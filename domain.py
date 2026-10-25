from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
OPTIONS_PATH = ROOT / 'data' / 'feature_options.json'
if OPTIONS_PATH.exists():
    OPTIONS = json.loads(OPTIONS_PATH.read_text(encoding='utf-8'))
else:
    OPTIONS = {'age':['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'], 'columns':{}, 'xgb_categories':{}}

MEDS = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide",
        "glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
        "insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
        "metformin-rosiglitazone","metformin-pioglitazone"]

AGE_OPTIONS = OPTIONS.get('age', [])
FIELD_LABELS = {
 'age':'Tranche d’âge','gender':'Sexe enregistré','race':'Groupe démographique','admission_type_id':'Code de type d’admission',
 'admission_source_id':'Code de provenance','medical_specialty':'Spécialité du service','payer_code':'Code payeur',
 'time_in_hospital':'Durée d’hospitalisation (jours)',
 'num_lab_procedures':'Nombre d’analyses de laboratoire','num_procedures':'Nombre d’actes / procédures',
 'num_medications':'Nombre de médicaments prescrits','number_outpatient':'Visites ambulatoires antérieures',
 'number_emergency':'Passages aux urgences antérieurs','number_inpatient':'Hospitalisations antérieures',
 'diag_1':'Diagnostic principal (code)','diag_2':'Diagnostic secondaire (code)','diag_3':'Troisième diagnostic (code)',
 'number_diagnoses':'Nombre de diagnostics consignés','max_glu_serum':'Dernier résultat de glycémie disponible',
 'A1Cresult':'Statut HbA1c consigné','change':'Modification du traitement durant le séjour',
 'diabetesMed':'Médicament antidiabétique consigné',
}
ORAL = [m for m in MEDS if m != 'insulin']
for _m in MEDS: FIELD_LABELS[_m] = _m.replace('-', ' ').capitalize()

NUMERIC_BOUNDS = {
 'time_in_hospital':(1,14,'1 à 14 jours'),
 'num_lab_procedures':(0,132,'0 à 132 examens'),
 'num_procedures':(0,6,'0 à 6 procédures'),
 'num_medications':(0,81,'0 à 81 médicaments'),
 'number_outpatient':(0,42,'Visites antérieures, 0 à 42'),
 'number_emergency':(0,76,'Passages antérieurs, 0 à 76'),
 'number_inpatient':(0,21,'Hospitalisations antérieures, 0 à 21'),
 'number_diagnoses':(1,16,'1 à 16 diagnostics'),
}

STEP_META = {
 1: {'title':'Profil du patient','subtitle':'Commencez par le contexte utile au modèle. Utilisez une référence pseudonymisée, jamais un nom complet.', 'icon':'user-round'},
 2: {'title':'Contexte d’admission','subtitle':'Renseignez les informations d’admission et le service sans inventer de codes.', 'icon':'hospital'},
 3: {'title':'Données cliniques','subtitle':'Les champs dépendent du module. Les codes sont assistés par recherche et validés à la saisie.', 'icon':'activity'},
 4: {'title':'Historique & traitements','subtitle':'Confirmez l’historique déclaré et les traitements visibles dans le dossier.', 'icon':'clipboard-list'},
 5: {'title':'Vérification finale','subtitle':'Relisez les valeurs avant le calcul. Vous pouvez revenir à chaque étape pour les corriger.', 'icon':'shield-check'},
 6: {'title':'Résultat & traçabilité','subtitle':'Résultat expérimental à confirmer par l’équipe médicale et les données sources.', 'icon':'chart-no-axes-combined'},
}

# Core model source fields for each workflow. Derived features are created server-side.
TASKS = {
 'long-stay': {
  'title':'Durée de séjour','short':'Séjour long','group':'Organisation des lits', 'icon':'calendar-clock', 'color':'cyan',
  'description':'Estimer la probabilité qu’un séjour atteigne 7 jours ou plus à partir du profil hospitalier et thérapeutique consigné.',
  'objective':'Anticipation de la charge de lits et coordination de sortie',
  'model':'XGBoost non équilibré','target':'Séjour ≥ 7 jours','kind':'binary','threshold':0.27,
  'positive':'Signal de séjour long','negative':'Signal de séjour plus court',
  'note':'Le seuil 0,27 provient d’une comparaison exploratoire. Plusieurs variables peuvent n’être connues qu’après l’admission. Pas d’usage clinique réel sans validation externe.',
  'fields':set(['age','gender','race','admission_type_id','admission_source_id','payer_code','medical_specialty',
    'num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient',
    'diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum','A1Cresult']+MEDS+['change','diabetesMed']),
  'stages':{1:['age','gender','race'],2:['admission_type_id','admission_source_id','medical_specialty','payer_code'],
   3:['num_lab_procedures','num_procedures','num_medications','number_diagnoses','diag_1','diag_2','diag_3','max_glu_serum','A1Cresult'],
   4:['number_outpatient','number_emergency','number_inpatient','change','diabetesMed']+MEDS},
 },
 'readmission': {
  'title':'Réadmission','short':'Réadmission','group':'Continuité des soins','icon':'repeat-2','color':'violet',
  'description':'Estimer le risque de réadmission avec un modèle XGBoost binaire (0 = pas de réadmission, 1 = réadmission) optimisé pour maximiser la précision.',
  'objective':'Préparer le suivi post-sortie et les appels de continuité', 'model':'XGBoost','target':'Prédiction binaire (0 ou 1)','kind':'binary','threshold':0.5,
  'positive':'Classe 1 — Réadmission','negative':'Classe 0 — Pas de réadmission',
  'note':'Modèle prédictif binaire XGBoost (classe 0 ou 1) entraîné pour optimiser la précision globale (~64%, ROC-AUC ~0.69). Les sorties assistent l’organisation et le suivi médical sans se substituer au jugement clinique.',
  'fields':set(['age','gender','race','admission_type_id','admission_source_id','medical_specialty',
                'diag_1','diag_2','diag_3','number_diagnoses','num_lab_procedures','num_procedures','num_medications',
                'number_outpatient','number_emergency','number_inpatient','time_in_hospital','change','diabetesMed','insulin','max_glu_serum','A1Cresult']),
  'stages':{
    1:['age','gender','race'],
    2:['admission_type_id','admission_source_id','medical_specialty'],
    3:['diag_1','diag_2','diag_3','number_diagnoses','num_lab_procedures','num_procedures','num_medications'],
    4:['number_outpatient','number_emergency','number_inpatient','time_in_hospital','change','diabetesMed','insulin','max_glu_serum','A1Cresult']
  },
 },
 'care-load': {
  'title':'Charge de soins','short':'Charge de soins','group':'Planification des ressources','icon':'gauge','color':'amber',
  'description':'Estimer si le dossier ressemble à un profil de forte charge construit à partir de volumes d’actes, de médicaments et d’utilisation hospitalière.',
  'objective':'Aider à la planification des ressources et au suivi prioritaire', 'model':'Régression logistique sur cible proxy K-Means','target':'Classe proxy de forte charge de soins','kind':'binary','threshold':0.5,
  'positive':'Profil proxy de forte charge','negative':'Profil proxy de charge plus modérée',
  'note':'La classe cible vient d’un clustering, pas d’un diagnostic clinique indépendant. C’est un outil de recherche pour l’organisation, non une mesure validée du besoin de soins.',
  'fields':set(['age','gender','race','admission_type_id','admission_source_id','medical_specialty','insulin','diabetesMed']),
  'stages':{1:['age','gender','race'],2:['admission_type_id','admission_source_id','medical_specialty'],3:[],4:['insulin','diabetesMed']},
 },
 'care-profile': {
  'title':'Profils de soins','short':'Profils de soins','group':'Personnalisation du parcours','icon':'layers-3','color':'emerald',
  'description':'Classer les informations de traitement, d’examens et d’utilisation en un profil de gestion des soins observé.',
  'objective':'Comparer les pratiques documentées et structurer l’analyse de parcours','model':'K-Means (k = 3)','target':'Profil de gestion observé','kind':'cluster','threshold':None,
  'positive':'','negative':'','note':'Le clustering décrit la gestion enregistrée, pas un type de diabète. Sa séparation est faible (silhouette de référence proche de 0,16), donc les profils se chevauchent.',
  'fields':set(['age','admission_type_id','admission_source_id','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses','A1Cresult','max_glu_serum','change']+MEDS),
  'stages':{1:['age'],2:['admission_type_id','admission_source_id'],3:['num_lab_procedures','num_procedures','num_medications','number_diagnoses','A1Cresult','max_glu_serum'],4:['number_outpatient','number_emergency','number_inpatient','change']+MEDS},
 },
 'hba1c': {
  'title':'Surveillance HbA1c','short':'Surveillance HbA1c','group':'Prévention et suivi','icon':'droplet','color':'rose',
  'description':'Classer le statut le plus probable du suivi HbA1c — absence de test, résultat normal ou résultat anormal — à partir du profil d’admission et des visites antérieures.',
  'objective':'Repérer les dossiers dont le suivi biologique mérite une vérification','model':'Régression logistique exploratoire','target':'None / Norm / Anormal','kind':'multiclass','threshold':None,
  'positive':'','negative':'','note':'Vérifiez toujours le résultat du laboratoire et les dates. Une classe “absence de test” signifie que le dataset historique ne documente pas de test; ce n’est pas une interprétation biologique.',
  'fields':set(['age','gender','race','admission_type_id','admission_source_id','medical_specialty','diag_1','number_outpatient','number_emergency','number_inpatient']),
  'stages':{1:['age','gender','race'],2:['admission_type_id','admission_source_id','medical_specialty'],3:['diag_1'],4:['number_outpatient','number_emergency','number_inpatient']},
 },
 'treatment-pattern': {
  'title':'Profil thérapeutique','short':'Profil thérapeutique','group':'Revue pharmaceutique','icon':'pill','color':'blue',
  'description':'Estimer un motif thérapeutique historique associé à la présence d’insuline et/ou de metformine, à partir d’autres informations du dossier.',
  'objective':'Soutenir l’analyse pharmaceutique et la revue de dossier','model':'Régression logistique exploratoire','target':'Motif historique de traitement','kind':'binary','threshold':0.5,
  'positive':'Motif associé à insuline et/ou metformine','negative':'Motif moins associé à ces traitements',
  'note':'Ce module ne recommande jamais un médicament et ne doit pas être utilisé pour initier, arrêter ou modifier un traitement.',
  'fields':set(['age','gender','race','admission_type_id','admission_source_id','diag_1','diag_2','diag_3','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses','repaglinide','nateglinide','chlorpropamide','glimepiride','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','tolazamide','glyburide-metformin','glipizide-metformin','change']),
  'stages':{1:['age','gender','race'],2:['admission_type_id','admission_source_id'],3:['diag_1','diag_2','diag_3','num_lab_procedures','num_procedures','number_diagnoses'],4:['num_medications','number_outpatient','number_emergency','number_inpatient','change','repaglinide','nateglinide','chlorpropamide','glimepiride','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','tolazamide','glyburide-metformin','glipizide-metformin']},
 },
}

STEP_TITLES = {1:'Profil patient',2:'Admission',3:'Clinique',4:'Historique',5:'Vérification',6:'Résultat'}


def field_definition(name: str) -> dict:
    data = {'name':name, 'label':FIELD_LABELS.get(name, name.replace('_',' ').capitalize()), 'required':True, 'hint':'', 'type':'select', 'options':OPTIONS.get('columns',{}).get(name,[])}
    if name == 'age':
        data.update(type='cards', options=AGE_OPTIONS, hint='Tranche déclarée dans les données historiques.')
    elif name in {'gender','race'}:
        data.update(type='cards', options=OPTIONS.get('columns',{}).get(name,[]), hint='Choisissez la valeur telle qu’elle figure au dossier.')
    elif name in {'diag_1','diag_2','diag_3'}:
        data.update(type='search', options=OPTIONS.get('columns',{}).get(name, OPTIONS.get('columns',{}).get('diag_1',[])), hint='Saisissez un code connu ou choisissez une suggestion; les codes non reconnus sont bloqués.')
    elif name in {'A1Cresult','max_glu_serum','change','diabetesMed'}:
        data.update(type='segmented', hint='Sélectionnez une seule valeur.')
    elif name in MEDS:
        data.update(type='medication', options=['Inconnu','No','Steady','Up','Down'], hint='Choisissez l’état consigné. « Non documenté » est différent de « Non ».')
    elif name in NUMERIC_BOUNDS:
        lo,hi,hint = NUMERIC_BOUNDS[name]
        data.update(type='number', min=lo, max=hi, hint=hint)
    elif name in {'admission_type_id','admission_source_id'}:
        data.update(type='select', options=OPTIONS.get('columns',{}).get(name,[]), hint='Utilisez le code tel qu’il figure dans le dossier.')
    elif name == 'medical_specialty':
        data.update(type='select', options=OPTIONS.get('columns',{}).get(name,[]), hint='Choisissez “Inconnu” si le service n’est pas documenté.')
    elif name == 'payer_code':
        data.update(type='select', options=OPTIONS.get('columns',{}).get(name,[]), hint='Choisissez “Inconnu” si non documenté.')
    return data


def get_task(task_id: str) -> dict | None:
    task = TASKS.get(task_id)
    if task is None: return None
    return {**task, 'id':task_id}


def task_fields_for_step(task_id: str, step: int) -> list[dict]:
    task = TASKS[task_id]
    names = [x for x in task['stages'].get(step, []) if x in task['fields']]
    # Keep a consistent schema but don't collect irrelevant data for the selected model.
    seen = set(); output=[]
    for name in names:
        if name not in seen:
            output.append(field_definition(name)); seen.add(name)
    return output


def all_task_fields(task_id: str) -> list[dict]:
    names=[]
    for step in range(1,5):
        names.extend([x['name'] for x in task_fields_for_step(task_id, step)])
    out=[]; seen=set()
    for name in names:
        if name not in seen: out.append(field_definition(name)); seen.add(name)
    return out


def get_med_fields(task_id: str) -> list[dict]:
    return [field_definition(m) for m in MEDS if m in TASKS[task_id]['fields']]
