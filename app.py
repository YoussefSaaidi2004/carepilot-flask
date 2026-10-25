from __future__ import annotations
import json, os, re, sqlite3, secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path

import click
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
from flask import Flask, abort, flash, g, jsonify, redirect, render_template, request, session, url_for
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash

from domain import ROOT, TASKS, OPTIONS, STEP_META, STEP_TITLES, field_definition, task_fields_for_step, all_task_fields
from ml_service import get_model_service, model_manifest

app = Flask(__name__, instance_relative_config=True)
Path(app.instance_path).mkdir(parents=True, exist_ok=True)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_urlsafe(48),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE', '0') == '1',
    PERMANENT_SESSION_LIFETIME=60*60*2,
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    WTF_CSRF_TIME_LIMIT=3600,
)
csrf = CSRFProtect(app)
DB_PATH = Path(app.instance_path) / 'carepilot.sqlite3'


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db():
    with connect_db() as conn:
        conn.executescript('''
          CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'medical_staff',
            created_at TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            patient_ref TEXT NOT NULL DEFAULT '',
            result_label TEXT NOT NULL,
            score_percent REAL,
            positive INTEGER,
            result_json TEXT NOT NULL,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(created_by) REFERENCES users(id)
          );
          CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
          );
          CREATE TABLE IF NOT EXISTS wizard_drafts (
            user_id INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY(user_id, task_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
          );
        ''')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def audit(action: str, detail: str = ''):
    user_id = session.get('user_id')
    with connect_db() as conn:
        conn.execute('INSERT INTO audit_log(user_id, action, detail, created_at) VALUES(?,?,?,?)',
                     (user_id, action, detail[:220], utc_now()))


def load_wizard(task_id: str):
    # Load transient form values from SQLite, never from a client-side cookie.
    user_id = session.get('user_id')
    if not user_id:
        return {}, {}
    cutoff = datetime.now(timezone.utc).timestamp() - 60 * 60
    with connect_db() as conn:
        row = conn.execute('SELECT payload_json, result_json, updated_at FROM wizard_drafts WHERE user_id=? AND task_id=?',
                           (user_id, task_id)).fetchone()
        if not row:
            return {}, {}
        try:
            updated = datetime.fromisoformat(row['updated_at']).timestamp()
        except (TypeError, ValueError):
            updated = 0
        if updated < cutoff:
            conn.execute('DELETE FROM wizard_drafts WHERE user_id=? AND task_id=?', (user_id, task_id))
            return {}, {}
        return json.loads(row['payload_json'] or '{}'), json.loads(row['result_json'] or '{}')


def save_wizard(task_id: str, payload: dict, result: dict | None = None):
    # Save a short-lived server-side draft; completed workflows discard raw form values.
    user_id = session.get('user_id')
    if not user_id:
        return
    statement = ("INSERT INTO wizard_drafts(user_id, task_id, payload_json, result_json, updated_at) "
                 "VALUES(?,?,?,?,?) ON CONFLICT(user_id, task_id) DO UPDATE SET "
                 "payload_json=excluded.payload_json, result_json=excluded.result_json, updated_at=excluded.updated_at")
    with connect_db() as conn:
        conn.execute(statement, (user_id, task_id, json.dumps(payload, ensure_ascii=False),
                                 json.dumps(result or {}, ensure_ascii=False), utc_now()))


def login_required(fn):
    return fn


def ensure_local_user():
    """Provide an internal owner for local records without requiring sign-in."""
    if session.get('user_id'):
        return
    with connect_db() as conn:
        user = conn.execute(
            'SELECT id, username FROM users ORDER BY id LIMIT 1'
        ).fetchone()
        if not user:
            cursor = conn.execute(
                'INSERT INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)',
                ('local', generate_password_hash(secrets.token_urlsafe(32)),
                 'local', utc_now()),
            )
            user = {'id': cursor.lastrowid, 'username': 'local'}
    session['user_id'] = user['id']
    session['username'] = user['username']


@app.before_request
def before_request():
    ensure_local_user()
    g.username = session.get('username', 'Espace local')
    # Automatically purge inactive drafts even if a user never revisits that specific module.
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec='seconds')
    with connect_db() as conn:
        conn.execute('DELETE FROM wizard_drafts WHERE updated_at < ?', (cutoff,))


@app.context_processor
def inject_globals():
    return {'nav_active': request.endpoint or '', 'current_user': session.get('username'),
            'task_catalog': [{**v, 'id':k, 'fields_count':len(v['fields'])} for k,v in TASKS.items()]}


@app.route('/health')
def health():
    return jsonify({'status':'ok'})


@app.route('/login', methods=['GET','POST'])
def login():
    return redirect(url_for('dashboard'))


@app.post('/logout')
@login_required
def logout():
    audit('LOGOUT', 'Déconnexion')
    with connect_db() as conn:
        conn.execute('DELETE FROM wizard_drafts WHERE user_id=?', (session['user_id'],))
    session.clear()
    flash('Vous êtes déconnecté.', 'success')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    with connect_db() as conn:
        total = conn.execute('SELECT COUNT(*) FROM assessments').fetchone()[0]
        today = conn.execute("SELECT COUNT(*) FROM assessments WHERE substr(created_at,1,10) = ?", (datetime.now().date().isoformat(),)).fetchone()[0]
        users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        recent = conn.execute('SELECT * FROM assessments ORDER BY id DESC LIMIT 6').fetchall()
        by_task = conn.execute('SELECT task_id, COUNT(*) AS n FROM assessments GROUP BY task_id').fetchall()
    counts = {r['task_id']:r['n'] for r in by_task}
    rows = []
    for key, task in TASKS.items():
        rows.append({'id':key,'title':task['short'],'icon':task['icon'],'color':task['color'],'count':counts.get(key,0),'group':task['group']})
    return render_template('dashboard.html', total=total, today=today, users=users, recent=recent, task_counts=rows,
                           task_lookup=TASKS)


@app.route('/tasks')
@login_required
def tasks_page():
    return render_template('tasks.html', tasks=[{**v,'id':k,'fields_count':len(v['fields'])} for k,v in TASKS.items()])


def validate_field(field: dict, raw: str) -> str:
    name=field['name']; raw=(raw or '').strip()
    if field.get('required', True) and raw == '':
        raise ValueError(f'Le champ « {field["label"]} » est obligatoire.')
    if raw == '': return ''
    if field['type'] == 'number':
        if raw == 'Inconnu':
            return 'Inconnu'
        try: value=float(raw)
        except ValueError: raise ValueError(f'« {field["label"]} » doit être un nombre.')
        if not value.is_integer(): raise ValueError(f'« {field["label"]} » doit être un entier.')
        value=int(value)
        if value < field['min'] or value > field['max']:
            raise ValueError(f'« {field["label"]} » doit être compris entre {field["min"]} et {field["max"]}.')
        return str(value)
    allowed = set(str(v) for v in field.get('options', []))
    if field['type'] == 'search':
        allowed = set(str(v) for v in OPTIONS.get('columns',{}).get(name,[]))
    if not allowed:
        raise ValueError(f'La liste autorisée pour « {field["label"]} » est indisponible. Contactez l’administrateur.')
    if raw not in allowed:
        raise ValueError(f'Valeur non autorisée pour « {field["label"]} ». Choisissez une option proposée.')
    return raw


def _patient_ref(raw: str) -> str:
    raw=(raw or '').strip()
    if not raw: return ''
    if len(raw)>32 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,31}', raw):
        raise ValueError('La référence doit contenir 1 à 32 caractères alphanumériques, point, tiret ou underscore. Utilisez une référence pseudonymisée.')
    return raw


def _save_assessment(task_id: str, payload: dict, result: dict) -> int:
    task=TASKS[task_id]
    patient_ref=payload.get('patient_ref','')
    score=result.get('score_percent')
    positive=result.get('positive')
    with connect_db() as conn:
        cur=conn.execute('''INSERT INTO assessments(task_id, patient_ref, result_label, score_percent, positive, result_json, created_by, created_at)
                           VALUES(?,?,?,?,?,?,?,?)''',
                         (task_id, patient_ref, result['label'], score, int(positive) if positive is not None else None,
                          json.dumps(result, ensure_ascii=False), session['user_id'], utc_now()))
        record_id=cur.lastrowid
    audit('PREDICTION', f'{task_id} · évaluation {record_id}')
    return int(record_id)


@app.route('/tasks/<task_id>')
@login_required
def start_task(task_id):
    if task_id not in TASKS: abort(404)
    with connect_db() as conn:
        conn.execute('DELETE FROM wizard_drafts WHERE user_id=? AND task_id=?', (session['user_id'], task_id))
    return redirect(url_for('task_step', task_id=task_id, step=1))


@app.route('/tasks/<task_id>/step/<int:step>', methods=['GET','POST'])
@login_required
def task_step(task_id, step):
    if task_id not in TASKS or step < 1 or step > 6: abort(404)
    task={**TASKS[task_id], 'id':task_id}
    payload, stored_result = load_wizard(task_id)
    payload = dict(payload)
    errors=[]
    fields=task_fields_for_step(task_id, step) if step <= 4 else []
    if request.method == 'POST':
        if step <= 4:
            new_values={}
            for field in fields:
                try:
                    new_values[field['name']]=validate_field(field, request.form.get(field['name'], ''))
                except ValueError as exc:
                    errors.append(str(exc))
            if step == 1:
                try: new_values['patient_ref']=_patient_ref(request.form.get('patient_ref',''))
                except ValueError as exc: errors.append(str(exc))
            if not errors:
                payload.update(new_values)
                save_wizard(task_id, payload, stored_result)
                return redirect(url_for('task_step', task_id=task_id, step=step+1))
            # Keep submitted values visible after a validation error, but never save unvalidated inputs.
            for field in fields:
                payload[field['name']] = request.form.get(field['name'], '')[:200]
            if step == 1:
                payload['patient_ref'] = request.form.get('patient_ref', '')[:32]
        elif step == 5:
            # Enforce complete server-side validation even if the browser is bypassed.
            for field in all_task_fields(task_id):
                if payload.get(field['name'],'') == '':
                    errors.append(f'Le champ « {field["label"]} » doit être renseigné à l’étape correspondante.')
            if not errors:
                try:
                    result=get_model_service().predict(task_id, payload)
                    record_id=_save_assessment(task_id,payload,result)
                    result['record_id']=record_id
                    # Drop raw form data as soon as inference completes.
                    save_wizard(task_id, {}, result)
                    return redirect(url_for('task_step', task_id=task_id, step=6))
                except Exception as exc:
                    app.logger.exception('Prediction failed for task %s', task_id)
                    errors.append('Le calcul n’a pas pu être exécuté. Vérifiez les données saisies ou contactez l’administrateur technique.')
        else:
            return redirect(url_for('task_step', task_id=task_id, step=6))
    if step == 6 and not stored_result:
        flash('Lancez d’abord l’analyse depuis l’étape de vérification.', 'info')
        return redirect(url_for('task_step', task_id=task_id, step=5))
    result=stored_result if step == 6 else {}
    review=[]
    if step == 5:
        for page_num in range(1,5):
            page_fields=task_fields_for_step(task_id,page_num)
            if page_fields:
                review.append({'step':page_num,'title':STEP_META[page_num]['title'],'fields':[
                    {'label':f['label'],'value':('Non documenté' if payload.get(f['name']) == 'Inconnu' else payload.get(f['name'],'—')),'name':f['name']} for f in page_fields]})
        if payload.get('patient_ref'):
            review.insert(0, {'step':0,'title':'Référence dossier','fields':[{'label':'Référence pseudonymisée','value':payload['patient_ref'],'name':'patient_ref'}]})
    step_meta=STEP_META[step]
    total_fields=len(all_task_fields(task_id))
    return render_template('task_step.html', task=task, step=step, step_meta=step_meta, fields=fields, payload=payload,
                           errors=errors, review=review, result=result, step_titles=STEP_TITLES,
                           total_fields=total_fields, options=OPTIONS)


@app.route('/patients')
@login_required
def patients():
    q=request.args.get('q','').strip()
    task_filter=request.args.get('task','').strip()
    sql='SELECT a.*, u.username FROM assessments a LEFT JOIN users u ON u.id=a.created_by WHERE 1=1'
    params=[]
    if q:
        sql+=' AND (a.patient_ref LIKE ? OR CAST(a.id AS TEXT) LIKE ? OR a.result_label LIKE ?)'
        params.extend([f'%{q}%',f'%{q}%',f'%{q}%'])
    if task_filter in TASKS:
        sql+=' AND a.task_id = ?'; params.append(task_filter)
    sql+=' ORDER BY a.id DESC LIMIT 200'
    with connect_db() as conn:
        records=conn.execute(sql,params).fetchall()
    return render_template('patients.html', records=records, task_filter=task_filter, q=q, tasks=TASKS)


@app.route('/assessments/<int:record_id>')
@login_required
def assessment_detail(record_id):
    with connect_db() as conn:
        record=conn.execute('SELECT a.*, u.username FROM assessments a LEFT JOIN users u ON u.id=a.created_by WHERE a.id=?',(record_id,)).fetchone()
    if not record: abort(404)
    result=json.loads(record['result_json'])
    return render_template('assessment_detail.html', record=record, result=result, task=TASKS.get(record['task_id'],{}))


@app.route('/reports')
@login_required
def reports():
    with connect_db() as conn:
        total=conn.execute('SELECT COUNT(*) FROM assessments').fetchone()[0]
        by_task=conn.execute('SELECT task_id, COUNT(*) AS n FROM assessments GROUP BY task_id').fetchall()
        by_day=conn.execute("SELECT substr(created_at,1,10) as day, COUNT(*) as n FROM assessments GROUP BY day ORDER BY day DESC LIMIT 14").fetchall()
        outcome=conn.execute('SELECT positive, COUNT(*) AS n FROM assessments WHERE positive IS NOT NULL GROUP BY positive').fetchall()
    counts={r['task_id']:r['n'] for r in by_task}
    max_count=max(counts.values(),default=1)
    chart=[{'id':k,'label':v['short'],'count':counts.get(k,0),'width':round(100*counts.get(k,0)/max_count) if max_count else 0} for k,v in TASKS.items()]
    days=list(reversed([dict(r) for r in by_day]))
    max_day=max((r['n'] for r in days),default=1)
    for d in days: d['height']=max(8,round(100*d['n']/max_day))
    outcomes={str(r['positive']):r['n'] for r in outcome}
    return render_template('reports.html', total=total, chart=chart, days=days, outcomes=outcomes)


@app.route('/audit')
@login_required
def audit_page():
    with connect_db() as conn:
        rows=conn.execute('SELECT a.*, u.username FROM audit_log a LEFT JOIN users u ON u.id=a.user_id ORDER BY a.id DESC LIMIT 100').fetchall()
    return render_template('audit.html', rows=rows)


@app.route('/settings')
@login_required
def settings():
    manifest=model_manifest()
    with connect_db() as conn:
        user_count=conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    return render_template('settings.html', manifest=manifest, user_count=user_count,
                           cookie_secure=app.config['SESSION_COOKIE_SECURE'], secret_configured=bool(os.environ.get('SECRET_KEY')))


@app.cli.command('create-admin')
def create_admin():
    """Create the first medical staff account, or add another local account."""
    username=click.prompt('Nom de compte (sans données personnelles)').strip()
    if not re.fullmatch(r'[A-Za-z0-9._-]{3,32}', username):
        raise click.ClickException('Utilisez 3 à 32 caractères : lettres, chiffres, point, tiret ou underscore.')
    password=click.prompt('Mot de passe', hide_input=True, confirmation_prompt=True)
    if len(password)<12:
        raise click.ClickException('Choisissez au moins 12 caractères.')
    try:
        with connect_db() as conn:
            conn.execute('INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)',
                         (username,generate_password_hash(password), 'medical_staff', utc_now()))
    except sqlite3.IntegrityError:
        raise click.ClickException('Ce nom de compte existe déjà.')
    click.echo(f'Compte {username} créé. Gardez le mot de passe en lieu sûr.')


@app.cli.command('list-users')
def list_users():
    """List usernames only (never password hashes)."""
    with connect_db() as conn:
        rows=conn.execute('SELECT username, role, created_at FROM users ORDER BY id').fetchall()
    for row in rows: click.echo(f"{row['username']} · {row['role']} · {row['created_at']}")


init_db()

if __name__ == '__main__':
    # For local development only. Use a production WSGI server and HTTPS in deployment.
    app.run(host=os.environ.get('HOST','127.0.0.1'), port=int(os.environ.get('PORT','5000')), debug=os.environ.get('FLASK_DEBUG','0')=='1')
