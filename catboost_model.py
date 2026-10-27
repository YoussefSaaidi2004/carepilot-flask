from __future__ import annotations
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

TE_COLS = ['diag_1_3', 'diag_2_3', 'diag_3_3', 'specialty_cat', 'discharge_cat']

class CatWrap(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=300, depth=6, learning_rate=0.06):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        cats = [c for c in TE_COLS if c in X.columns]
        from catboost import CatBoostClassifier
        self.m_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            auto_class_weights='Balanced', cat_features=cats, verbose=0, thread_count=2,
            random_seed=42, allow_writing_files=False)
        self.m_.fit(X, self.le_.transform(y))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TwoStageClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, stage1=None, stage2=None):
        self.stage1 = stage1
        self.stage2 = stage2

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y) # ['Anormal', 'None', 'Norm']
        assert len(self.classes_) == 3, f"Expected 3 classes, got {self.classes_}"
        c_anormal, c_none, c_norm = self.classes_
        tested = (y != c_none)
        self.s1_ = clone(self.stage1).fit(X, tested.astype(int))
        self.s2_ = clone(self.stage2).fit(X[tested], (y[tested] == c_anormal).astype(int))
        return self

    def predict_proba(self, X):
        p_tested = self.s1_.predict_proba(X)[:, 1]
        q = self.s2_.predict_proba(X)[:, 1]
        return np.column_stack([p_tested * q, 1 - p_tested, p_tested * (1 - q)])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def group_diag(code):
    if code is None or pd.isna(code) or str(code).strip() in ['?', '', 'None', 'Missing']:
        return 'Missing'
    code_s = str(code).strip()
    try:
        code_f = float(code_s)
    except ValueError:
        return 'Autre'
    if 390 <= code_f <= 459 or code_f == 785:
        return 'Circulatoire'
    elif 250 <= code_f < 251:
        return 'Diabete'
    elif 460 <= code_f <= 519 or code_f == 786:
        return 'Respiratoire'
    elif 520 <= code_f <= 579 or code_f == 787:
        return 'Digestif'
    elif 800 <= code_f <= 999:
        return 'Traumatisme'
    elif 710 <= code_f <= 739:
        return 'Musculo_squelettique'
    elif 580 <= code_f <= 629:
        return 'Genito_urinaire'
    elif 140 <= code_f <= 239:
        return 'Neoplasme'
    else:
        return 'Autre'


def clean_col_name(c: str) -> str:
    return re.sub(r'[^0-9a-zA-Z_]+', '_', c)


def prepare_patient_features(payload: dict, artifact: dict) -> pd.DataFrame:
    medians = artifact.get('medians', {})
    modes = artifact.get('modes', {})
    age_map = artifact.get('age_map', {})

    def _num(key, default=0.0):
        val = payload.get(key)
        if val is None or val == '' or str(val).strip() in ['?', 'Inconnu', 'None']:
            return float(medians.get(key, default))
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(medians.get(key, default))

    def _cat(key, default='Missing'):
        val = payload.get(key)
        if val is None or val == '' or str(val).strip() in ['?', 'Inconnu', 'None']:
            return str(modes.get(key, default))
        return str(val)

    # 1. Base values
    time_in_hospital = _num('time_in_hospital', 4.0)
    num_lab_procedures = _num('num_lab_procedures', 44.0)
    num_procedures = _num('num_procedures', 1.0)
    num_medications = _num('num_medications', 15.0)
    number_outpatient = _num('number_outpatient', 0.0)
    number_emergency = _num('number_emergency', 0.0)
    number_inpatient = _num('number_inpatient', 0.0)
    number_diagnoses = _num('number_diagnoses', 8.0)

    age_val = _cat('age', '[50-60)')
    age_encoded = float(age_map.get(age_val, 5))

    admission_type_id = _cat('admission_type_id', '1')
    admission_source_id = _cat('admission_source_id', '7')
    discharge_disposition_id = _cat('discharge_disposition_id', '1')
    medical_specialty = _cat('medical_specialty', 'Missing')

    diag_1 = _cat('diag_1', 'Missing')
    diag_2 = _cat('diag_2', 'Missing')
    diag_3 = _cat('diag_3', 'Missing')

    grp1 = group_diag(diag_1)
    grp2 = group_diag(diag_2)
    grp3 = group_diag(diag_3)

    race = _cat('race', modes.get('race', 'Caucasian'))
    gender = _cat('gender', 'Female')
    max_glu_serum = _cat('max_glu_serum', 'None')
    change = _cat('change', 'No')
    diabetesMed = _cat('diabetesMed', 'Yes')
    insulin = _cat('insulin', 'No')

    # 2. Scaled numeric values
    num_df = pd.DataFrame([[
        num_lab_procedures, num_procedures, num_medications,
        number_outpatient, number_emergency, number_inpatient,
        number_diagnoses, time_in_hospital
    ]], columns=artifact['num_cols'], dtype=float)
    scaled_num = artifact['scaler'].transform(num_df)[0]

    # 3. Engineered features
    days = max(1.0, time_in_hospital)
    meds_per_day = num_medications / days
    labs_per_day = num_lab_procedures / days
    diag_per_day = number_diagnoses / days

    med_cols_raw = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
                    'glipizide','glyburide','pioglitazone','rosiglitazone','insulin']
    nb_meds_ch = sum(payload.get(m, 'No') not in ['No', '', '?', 'Inconnu'] for m in med_cols_raw)
    nb_meds_dose_changed = sum(payload.get(m, 'No') in ['Up', 'Down'] for m in med_cols_raw)
    insulin_dose_changed = int(insulin in ['Up', 'Down'])

    diag_grps = [grp1, grp2, grp3]
    nb_diag_diabetes = sum(g == 'Diabete' for g in diag_grps)
    nb_diag_groups = len(set(diag_grps))

    total_visits = number_outpatient + number_emergency + number_inpatient
    has_prior_inpatient = int(number_inpatient > 0)
    high_utilizer = int(total_visits >= 3)
    emergency_admission = int(str(admission_type_id) == '1')
    age_senior = int(age_encoded >= 7)
    polypharmacy = int(num_medications >= 15)

    scaled_total_visits = float(artifact['visit_scaler'].transform([[total_visits]])[0][0])
    scaled_nb_meds_ch = float(artifact['med_ch_scaler'].transform([[nb_meds_ch]])[0][0])

    # 4. Construct dummy row
    row = {col: 0.0 for col in artifact['dummy_columns']}

    # Put scaled numeric columns
    for col_name, val in zip(artifact['num_cols'], scaled_num):
        clean_c = clean_col_name(col_name)
        if clean_c in row:
            row[clean_c] = float(val)

    # Put encoded & engineered numericals
    row['age_encoded'] = float(age_encoded)
    row['total_visits_prior'] = scaled_total_visits
    row['nb_meds_changed'] = scaled_nb_meds_ch
    row['meds_per_day'] = float(meds_per_day)
    row['labs_per_day'] = float(labs_per_day)
    row['diag_per_day'] = float(diag_per_day)
    row['nb_meds_dose_changed'] = float(nb_meds_dose_changed)
    row['insulin_dose_changed'] = float(insulin_dose_changed)
    row['nb_diag_diabetes'] = float(nb_diag_diabetes)
    row['nb_diag_groups'] = float(nb_diag_groups)
    row['has_prior_inpatient'] = float(has_prior_inpatient)
    row['high_utilizer'] = float(high_utilizer)
    row['emergency_admission'] = float(emergency_admission)
    row['age_senior'] = float(age_senior)
    row['polypharmacy'] = float(polypharmacy)

    # Put active dummy categories
    active_dummies = [
        f"race_{race}", f"gender_{gender}", f"medical_specialty_{medical_specialty}",
        f"diag_1_group_{grp1}", f"diag_2_group_{grp2}", f"diag_3_group_{grp3}",
        f"max_glu_serum_{max_glu_serum}", f"change_{change}", f"diabetesMed_{diabetesMed}"
    ]
    for m in med_cols_raw:
        val = payload.get(m, 'No')
        if val not in ['No', '', '?', 'Inconnu']:
            active_dummies.append(f"{m}_{val}")

    for dummy_name in active_dummies:
        clean_name = clean_col_name(dummy_name)
        if clean_name in row:
            row[clean_name] = 1.0

    df_encoded = pd.DataFrame([row], columns=artifact['dummy_columns'])

    # 5. Raw categories for CatBoost
    raw_cat = pd.DataFrame([{
        'diag_1_3': str(diag_1)[:3],
        'diag_2_3': str(diag_2)[:3],
        'diag_3_3': str(diag_3)[:3],
        'specialty_cat': str(medical_specialty),
        'discharge_cat': str(discharge_disposition_id),
    }])

    df_final = pd.concat([df_encoded, raw_cat], axis=1)
    return df_final[artifact['feature_names']]


FEATURE_LABELS_FR = {
    'insulin_Up': "Ajustement à la hausse de la dose d'insuline",
    'insulin_Down': "Ajustement à la baisse de la dose d'insuline",
    'insulin_Steady': "Traitement insulinique en cours (stable)",
    'insulin_dose_changed': "Modification récente du schéma d'insuline",
    'nb_diag_diabetes': "Diagnostic principal ou secondaire codé diabète",
    'diag_1_3': "Diagnostic principal d'admission",
    'diag_2_3': "Comorbidité ou diagnostic associé",
    'specialty_cat': "Spécialité médicale du service hospitalier",
    'discharge_cat': "Mode d'orientation ou de sortie",
    'change_Ch': "Changement de molécule antidiabétique durant le séjour",
    'change_No': "Aucun changement thérapeutique documenté",
    'diabetesMed_Yes': "Médication antidiabétique active",
    'diabetesMed_No': "Absence de traitement antidiabétique consigné",
    'max_glu_serum_None': "Glycémie sérique non mesurée",
    'max_glu_serum_300': "Glycémie sérique très élevée (> 300 mg/dL)",
    'max_glu_serum_200': "Glycémie sérique élevée (> 200 mg/dL)",
    'num_medications': "Nombre élevé de médicaments prescrits",
    'num_lab_procedures': "Volume d'analyses biologiques réalisées",
    'time_in_hospital': "Durée de séjour hospitalier prolongée",
    'number_diagnoses': "Complexité du profil (nombre de diagnostics)",
    'labs_per_day': "Intensité journalière de surveillance biologique",
    'meds_per_day': "Nombre de médicaments par jour d'hospitalisation",
    'has_prior_inpatient': "Antécédent d'hospitalisation l'année précédente",
    'high_utilizer': "Patient grand consommateur de soins (≥ 3 visites)",
    'emergency_admission': "Admission effectuée en contexte d'urgence",
    'age_senior': "Profil senior (âge ≥ 70 ans)",
    'polypharmacy': "Polymédication documentée (≥ 15 médicaments)",
    'total_visits_prior': "Fréquentation hospitalière antérieure élevée",
    'number_emergency': "Passages antérieurs aux urgences",
    'number_inpatient': "Nombre d'hospitalisations antérieures",
    'number_outpatient': "Consultations ambulatoires préalables",
    'metformin_Steady': "Prise régulière de metformine",
    'metformin_Up': "Augmentation de la dose de metformine",
    'glipizide_Up': "Augmentation de la dose de glipizide",
    'glimepiride_Up': "Augmentation de la dose de glimepiride"
}


def explain_patient_prediction(X: pd.DataFrame, model, pred: str, q_anormal: float, p_tested: float, payload: dict = None) -> dict:
    from catboost import Pool
    cats = [c for c in TE_COLS if c in X.columns]
    pool = Pool(X, cat_features=cats)

    if pred == 'Anormal' or q_anormal >= 0.40:
        raw_shaps = model.s2_.m_.get_feature_importance(pool, type='ShapValues')[0][:-1]
        target_name = "Risque d'HbA1c anormale (> 7%)"
    else:
        raw_shaps = model.s1_.m_.get_feature_importance(pool, type='ShapValues')[0][:-1]
        target_name = "Probabilité de prescription d'HbA1c"

    contribs = pd.Series(raw_shaps, index=X.columns)
    active_cols = [c for c in X.columns if (X[c].iloc[0] != 0 and X[c].iloc[0] != '0') or c in TE_COLS or c in ['num_lab_procedures', 'num_medications', 'time_in_hospital', 'number_diagnoses']]
    filtered = contribs.loc[contribs.index.intersection(active_cols)]
    if len(filtered) < 4:
        filtered = contribs

    top_pos = filtered.sort_values(ascending=False).head(3)
    top_neg = filtered.sort_values(ascending=True).head(2)

    factors = []
    max_val = max(float(abs(filtered).max()), 0.01)

    for col, val in top_pos.items():
        if val > 0.005:
            pct = int(round((float(val) / max_val) * 100))
            factors.append({
                'label': FEATURE_LABELS_FR.get(col, col.replace('_', ' ').capitalize()),
                'type': 'risk',
                'impact': f"+{pct}%",
                'weight': min(pct, 100)
            })

    for col, val in top_neg.items():
        if val < -0.005:
            pct = int(round((abs(float(val)) / max_val) * 100))
            factors.append({
                'label': FEATURE_LABELS_FR.get(col, col.replace('_', ' ').capitalize()),
                'type': 'protective',
                'impact': f"-{pct}%",
                'weight': min(pct, 100)
            })

    if pred == 'Anormal' or q_anormal >= 0.50:
        priority = {
            'level': 'danger',
            'badge': 'Priorité Élevée · Alerte Déséquilibre Glycémique',
            'color': '#be123c',
            'bg': 'rgba(225, 29, 72, 0.08)',
            'border': '#f43f5e'
        }
        actions = [
            "Surveillance renforcée des glycémies capillaires (dextro pré et post-prandiaux).",
            "Réévaluation de l'insulinothérapie ou de l'antidiabétique oral en cours.",
            "Demande d'avis spécialisé en diabétologie avant l'autorisation de sortie.",
            "Consultation d'éducation thérapeutique (observance et prévention des complications)."
        ]
    elif pred == 'None':
        priority = {
            'level': 'warning',
            'badge': 'Priorité Modérée · Alerte Dépistage (Protocole ADA)',
            'color': '#b45309',
            'bg': 'rgba(245, 158, 11, 0.08)',
            'border': '#f59e0b'
        }
        actions = [
            "Vérifier dans le dossier médical si un dosage d'HbA1c date de moins de 90 jours.",
            "Prescrire un dosage d'HbA1c veineuse à jeun selon le protocole institutionnel ADA.",
            "Consigner le statut glycémique dans la feuille de synthèse d'admission."
        ]
    else:
        priority = {
            'level': 'success',
            'badge': 'Priorité Standard · Contrôle Glycémique Satisfaisant',
            'color': '#15803d',
            'bg': 'rgba(34, 197, 94, 0.08)',
            'border': '#22c55e'
        }
        actions = [
            "Maintien du traitement antidiabétique actuel sans modification requise.",
            "Surveillance glycémique standard selon le protocole de routine du service.",
            "Planification du prochain contrôle d'HbA1c à 3 ou 6 mois en suivi ambulatoire."
        ]

    # Analyse de Pharmacovigilance & Sécurité Iatrogène
    payload = payload or {}
    pharmaco_alerts = []
    try:
        num_meds = int(payload.get('num_medications') or 0)
    except (ValueError, TypeError):
        num_meds = 0

    has_insulin = payload.get('insulin', 'No') not in {'No', 'Inconnu'}
    has_metformin = payload.get('metformin', 'No') not in {'No', 'Inconnu'}
    has_change = payload.get('change') == 'Ch'
    is_elderly = payload.get('age') in {'[70-80)', '[80-90)', '[90-100)'}

    if has_insulin and (num_meds >= 10 or is_elderly):
        pharmaco_alerts.append({
            'badge': 'Risque Hypoglycémique Iatrogène',
            'severity': 'danger',
            'color': '#be123c',
            'bg': '#fff1f2',
            'border': '#fecdd3',
            'title': 'Alerte Hypoglycémie Sévère sous Insuline',
            'desc': "Patient sous insuline avec polymédication/sujet âgé : risque élevé d'hypoglycémies asymptomatiques. Protocoliser un contrôle glycémique nocturne (3h00) et prescrire un kit de glucagon d'urgence avec protocole de resucrage (15g de glucides rapides)."
        })

    if has_metformin and (is_elderly or num_meds >= 12):
        pharmaco_alerts.append({
            'badge': 'Sécurité Rénale & Metformine',
            'severity': 'warning',
            'color': '#b45309',
            'bg': '#fffbeb',
            'border': '#fde68a',
            'title': 'Contrôle Fonction Rénale & Metformine',
            'desc': "Sous metformine chez sujet âgé ou polymédiqué : risque d'accumulation et d'acidose lactique. Contrôle systématique de la créatininémie et calcul du DFG. Suspension temporaire obligatoire 48h avant tout examen radiologique avec injection de produit de contraste iodé."
        })

    if has_change:
        pharmaco_alerts.append({
            'badge': 'Cinétique & Transition',
            'severity': 'info',
            'color': '#0284c7',
            'bg': '#f0f9ff',
            'border': '#bae6fd',
            'title': 'Instabilité Thérapeutique Récemment Consignée',
            'desc': "Ajustement posologique récent consigné au dossier : période de vulnérabilité accrue. Réaliser un profil glycémique en 6 points (pré et post-prandiaux) pour valider l'équilibre avant l'autorisation de sortie."
        })

    if num_meds >= 15:
        pharmaco_alerts.append({
            'badge': 'Polymédication Majeure',
            'severity': 'warning',
            'color': '#b45309',
            'bg': '#fffbeb',
            'border': '#fde68a',
            'title': f'Revue d\'Ordonnance Recommandée ({num_meds} molécules)',
            'desc': "Nombre très élevé de molécules concourantes : risque accru d'interactions médicamenteuses interférant avec les antidiabétiques. Une conciliation médicamenteuse par l'équipe de pharmacie clinique est préconisée."
        })

    if not pharmaco_alerts:
        pharmaco_alerts.append({
            'badge': 'Sécurité Thérapeutique',
            'severity': 'success',
            'color': '#15803d',
            'bg': '#f0fdf4',
            'border': '#bbf7d0',
            'title': 'Profil Iatrogène Sans Alerte Critique Immédiate',
            'desc': "Aucune interaction médicamenteuse critique détectée dans les données d'admission. Poursuivre la surveillance biologique usuelle selon les protocoles de routine du service."
        })

    return {
        'target_name': target_name,
        'factors': factors[:5],
        'priority': priority,
        'actions': actions,
        'pharmaco_alerts': pharmaco_alerts
    }
