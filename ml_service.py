from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import json, re
import joblib
import numpy as np
import pandas as pd
from domain import ROOT, OPTIONS, MEDS, ORAL, TASKS

MODEL_DIR = ROOT / 'models'

class ModelService:
    def __init__(self):
        self.xgb = joblib.load(MODEL_DIR / 'modeles_comparaison.joblib')
        self.readmission = joblib.load(MODEL_DIR / 'readmission.joblib')
        self.care_load = joblib.load(MODEL_DIR / 'care_load.joblib')
        self.care_profiles = joblib.load(MODEL_DIR / 'care_profiles.joblib')
        self.hba1c = joblib.load(MODEL_DIR / 'hba1c_status.joblib')
        self.treatment = joblib.load(MODEL_DIR / 'treatment_pattern.joblib')

    @staticmethod
    def _age_mid(value: str) -> float:
        match = re.search(r'\[(\d+)-(\d+)\)', str(value))
        if not match: raise ValueError('Tranche d’âge invalide.')
        return (int(match.group(1)) + int(match.group(2))) / 2.0

    @staticmethod
    def _numeric(payload, key):
        value = payload.get(key)
        if value == 'Inconnu':
            return float('nan')
        try: return float(value)
        except (KeyError, TypeError, ValueError): raise ValueError(f'Valeur numérique manquante ou invalide : {key}')

    @staticmethod
    def _profile_numeric(payload, key):
        value = ModelService._numeric(payload, key)
        if np.isfinite(value):
            return float(value)
        # The clustering notebook imputes missing count features before transforming them.
        imputation = OPTIONS.get('numeric_imputation', {}).get(key, 0.0)
        return float(imputation)

    @staticmethod
    def _cat(payload, key):
        value = payload.get(key, 'Inconnu')
        return str(value) if value not in (None, '') else 'Inconnu'

    def predict(self, task_id: str, payload: dict) -> dict:
        if task_id == 'long-stay': return self._predict_long_stay(payload)
        if task_id == 'readmission': return self._predict_readmission(payload)
        if task_id == 'care-load': return self._predict_care_load(payload)
        if task_id == 'care-profile': return self._predict_care_profile(payload)
        if task_id == 'hba1c': return self._predict_hba1c(payload)
        if task_id == 'treatment-pattern': return self._predict_treatment(payload)
        raise ValueError('Module de prédiction inconnu.')

    def _predict_long_stay(self, payload):
        feature_columns = self.xgb['feature_columns']
        numeric_columns = set(self.xgb['numeric_columns'])
        categorical_columns = set(self.xgb['categorical_columns'])
        row = {}
        for name in feature_columns:
            if name == 'age_num':
                row[name] = self._age_mid(payload['age'])
            elif name in numeric_columns:
                row[name] = self._numeric(payload, name)
            elif name in categorical_columns:
                value = self._cat(payload, name)
                allowed = OPTIONS.get('xgb_categories', {}).get(name, [])
                if value not in allowed: value = 'Inconnu'
                categories = allowed or [value]
                row[name] = pd.Categorical([value], categories=categories)
        X = pd.DataFrame({k: [v] if not isinstance(v, pd.Categorical) else v for k,v in row.items()})
        X = X[feature_columns]
        model = self.xgb['xgboost_unbalanced']
        score = float(model.predict_proba(X)[:,1][0])
        threshold = float(self.xgb.get('threshold_reference', 0.27))
        positive = score >= threshold
        return {
            'kind':'binary','label':TASKS['long-stay']['positive'] if positive else TASKS['long-stay']['negative'],
            'score':score,'score_percent':round(score*100,1),'threshold':threshold,'positive':positive,
            'score_label':'Score estimé par le modèle','threshold_label':f'Seuil expérimental {threshold:.2f}',
            'detail':'Le score n’est pas une probabilité clinique calibrée. Le modèle historique utilise aussi certaines variables qui peuvent se préciser après l’admission.',
            'subscores':[]
        }

    @staticmethod
    def _map_icd9_group(val: str) -> str:
        s = str(val).strip()
        if s.startswith('250'):
            return 'Diabetes'
        try:
            num = float(s)
            if (390 <= num <= 459) or num == 785: return 'Circulatory'
            if (460 <= num <= 519) or num == 786: return 'Respiratory'
            if (520 <= num <= 579) or num == 787: return 'Digestive'
            if (580 <= num <= 629) or num == 788: return 'Genitourinary'
            if 140 <= num <= 239: return 'Neoplasms'
            if 710 <= num <= 739: return 'Musculoskeletal'
            if 800 <= num <= 999: return 'Injury'
            return 'Other'
        except (ValueError, TypeError):
            return 'Other'

    def _prepare_readmission_row(self, payload: dict) -> pd.DataFrame:
        row = {}
        age_str = self._cat(payload, 'age')
        try:
            row['age_num'] = [self._age_mid(age_str)]
        except Exception:
            row['age_num'] = [65.0]

        num_keys = ['number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient',
                    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications']
        for k in num_keys:
            if k in payload and payload.get(k) not in (None, '', 'Inconnu'):
                try:
                    row[k] = [self._numeric(payload, k)]
                except Exception:
                    row[k] = [float('nan')]
            else:
                row[k] = [float('nan')]

        outp = 0.0 if np.isnan(row['number_outpatient'][0]) else float(row['number_outpatient'][0])
        emg = 0.0 if np.isnan(row['number_emergency'][0]) else float(row['number_emergency'][0])
        inp = 0.0 if np.isnan(row['number_inpatient'][0]) else float(row['number_inpatient'][0])
        tot = outp + emg + inp
        row['total_prior_visits'] = [tot]
        row['has_prior_inpatient'] = [int(inp > 0)]
        row['has_prior_emergency'] = [int(emg > 0)]
        row['inpatient_ratio'] = [inp / (tot + 1.0)]
        row['emergency_ratio'] = [emg / (tot + 1.0)]

        cat_keys = ['gender', 'race', 'admission_type_id', 'admission_source_id', 'medical_specialty',
                    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'insulin']
        for k in cat_keys:
            row[k] = [self._cat(payload, k)]

        row['diag_1_group'] = [self._map_icd9_group(self._cat(payload, 'diag_1'))]
        row['diag_2_group'] = [self._map_icd9_group(self._cat(payload, 'diag_2'))]
        row['diag_3_group'] = [self._map_icd9_group(self._cat(payload, 'diag_3'))]

        return pd.DataFrame(row)

    def _predict_readmission(self, payload):
        artifact = self.readmission
        pipeline = artifact['pipeline']

        if 'numeric_features' in artifact:
            X = self._prepare_readmission_row(payload)
        else:
            features = artifact['features']
            row = {}
            for f in features:
                if f in {'number_diagnoses','number_outpatient','number_emergency','number_inpatient'}:
                    row[f] = self._numeric(payload, f)
                else:
                    row[f] = self._cat(payload, f)
            X = pd.DataFrame([row], columns=features)

        proba = pipeline.predict_proba(X)[0]
        classes = list(pipeline.classes_)
        score = float(proba[classes.index(1)])
        threshold = float(artifact.get('threshold', 0.50))
        positive = score >= threshold
        binary_val = 1 if positive else 0
        binary_text = "Classe 1 (Réadmission)" if positive else "Classe 0 (Pas de réadmission)"
        acc = artifact.get('accuracy', 63.8)
        acc_str = f' (Précision ~{acc}%)' if acc else ''
        return {
            'kind': 'binary',
            'label': f'Classe {binary_val} — {"Réadmission" if positive else "Pas de réadmission"}',
            'binary_value': binary_val,
            'binary_label': binary_text,
            'score': score,
            'score_percent': round(score * 100, 1),
            'threshold': threshold,
            'positive': positive,
            'score_label': 'Probabilité estimée (XGBoost)',
            'threshold_label': f'Seuil {threshold:.2f} | Sortie binaire : {binary_val}{acc_str}',
            'detail': f'Classifieur binaire XGBoost (0 ou 1) optimisé pour la précision (Accuracy test: ~{acc}%, ROC-AUC: ~{artifact.get("roc_auc", 0.69):.2f}). Résultat binaire : {binary_val}.',
            'subscores': []
        }

    def _predict_care_load(self, payload):
        artifact=self.care_load; features=artifact['features']
        row={f:self._cat(payload,f) for f in features}
        X=pd.DataFrame([row],columns=features)
        proba=artifact['pipeline'].predict_proba(X)[0]
        classes=list(artifact['pipeline'].classes_)
        score=float(proba[classes.index(1)])
        positive=score >= float(artifact.get('threshold',0.5))
        return {'kind':'binary','label':artifact['positive_label'] if positive else artifact['negative_label'],
                'score':score,'score_percent':round(score*100,1),'threshold':float(artifact.get('threshold',0.5)),
                'positive':positive,'score_label':'Score du modèle proxy','threshold_label':'Seuil de démonstration 0,50',
                'detail':'La classe de référence a été dérivée d’un clustering K-Means sur la consommation de soins, pas d’un critère clinique validé.'}

    def _profile_row(self, payload):
        d={}
        d['age_mid']=self._age_mid(payload['age'])
        for c in ['num_lab_procedures','num_medications','number_diagnoses']:
            d[c]=self._profile_numeric(payload,c)
        for c in ['num_procedures','number_outpatient','number_emergency','number_inpatient']:
            d[c+'_log']=float(np.log1p(self._profile_numeric(payload,c)))
        meds={m:self._cat(payload,m) for m in MEDS}
        documented_used = {'Steady','Up','Down'}
        d['n_diab_meds']=sum(v in documented_used for v in meds.values())
        d['n_med_changes']=sum(v in {'Up','Down'} for v in meds.values())
        d['insulin_used']=int(meds['insulin'] in documented_used)
        d['insulin_changed']=int(meds['insulin'] in {'Up','Down'})
        d['oral_used']=int(any(meds[m] in documented_used for m in ORAL))
        d['med_change']=int(self._cat(payload,'change')=='Ch')
        a1c=self._cat(payload,'A1Cresult'); glu=self._cat(payload,'max_glu_serum')
        d['a1c_tested']=int(a1c in {'Norm','>7','>8'}); d['a1c_high']=int(a1c in {'>7','>8'})
        d['glu_tested']=int(glu in {'Norm','>200','>300'}); d['glu_high']=int(glu in {'>200','>300'})
        d['adm_emergency']=int(self._cat(payload,'admission_type_id') in {'1','2'})
        d['adm_er_source']=int(self._cat(payload,'admission_source_id')=='7')
        return d

    def _predict_care_profile(self,payload):
        artifact=self.care_profiles; features=artifact['features']
        X=pd.DataFrame([self._profile_row(payload)],columns=features).astype(float)
        z=artifact['scaler'].transform(X)
        cluster=int(artifact['kmeans'].predict(z)[0])
        distances=artifact['kmeans'].transform(z)[0]
        ordered=np.sort(distances)
        margin=float((ordered[1]-ordered[0])/(ordered[1]+1e-9)) if len(ordered)>1 else 0.0
        return {'kind':'cluster','label':artifact['names'].get(cluster,f'Profil {cluster+1}'),
                'cluster':cluster,'score':None,'score_percent':None,'distance':round(float(distances[cluster]),3),
                'proximity_percent':round(float(100*margin/(1+margin)),1),
                'score_label':'Proximité relative entre profils','threshold':None,
                'detail':'Une séparation faible entre clusters est attendue dans ce dataset. Le profil décrit des habitudes de gestion observées, pas un diagnostic ou un type de diabète.'}

    def _predict_hba1c(self, payload):
        artifact = self.hba1c
        if 'model' in artifact:
            from catboost_model import prepare_patient_features
            X = prepare_patient_features(payload, artifact)
            model = artifact['model']
            probs = model.predict_proba(X)[0]
            classes = list(model.classes_)
            index = int(np.argmax(probs))
            pred = str(classes[index])
            class_labels = artifact.get('class_labels', {})
            display = class_labels.get(pred, pred)

            p_tested = float(model.s1_.predict_proba(X)[0][1])
            q_anormal = float(model.s2_.predict_proba(X)[0][1])

            if pred == 'None':
                clinical_action = (
                    "⚠️ Alerte Dépistage (Protocole ADA) : Aucun test récent documenté. "
                    "Recommandation : prescrire un dosage d'HbA1c à jeun lors du bilan d'admission."
                )
            elif pred == 'Anormal':
                clinical_action = (
                    "🚨 Alerte Contrôle Glycémique : Risque élevé d'HbA1c anormale (> 7% ou > 8%). "
                    "Recommandation : surveillance rapprochée des glycémies capillaires et avis diabétologique pour réévaluation thérapeutique."
                )
            else:
                clinical_action = (
                    "✅ Profil Équilibré : Contrôle glycémique satisfaisant documenté (< 7%). "
                    "Poursuivre la surveillance selon le calendrier habituel du patient."
                )

            subscores = [
                {'label': class_labels.get(str(c), str(c)), 'value': round(float(p) * 100, 1)}
                for c, p in sorted(zip(classes, probs), key=lambda z: z[1], reverse=True)
            ]
            perf_note = f"Modèle séquentiel à deux étages · F1 macro {artifact.get('f1_macro', 0.482)} · ROC-AUC {artifact.get('roc_auc', 0.767)} · Précision globale {artifact.get('accuracy', 74.9)}%."
            detail = f"{perf_note} {clinical_action}"

            from catboost_model import explain_patient_prediction
            patient_context = {
                'patient_ref': payload.get('patient_ref') or 'DOS-0427',
                'age': payload.get('age', '[60-70)'),
                'gender': payload.get('gender', 'Féminin'),
                'insulin': payload.get('insulin', 'No'),
                'metformin': payload.get('metformin', 'No'),
                'num_medications': payload.get('num_medications', '15'),
                'num_lab_procedures': payload.get('num_lab_procedures', '44'),
                'time_in_hospital': payload.get('time_in_hospital', '4'),
                'change': payload.get('change', 'No'),
                'base_anormal_pct': round(q_anormal * 100, 1),
                'base_tested_pct': round(p_tested * 100, 1),
                'pred': pred
            }

            explanation = explain_patient_prediction(X, model, pred, q_anormal, p_tested, payload=payload)

            return {
                'kind': 'multiclass',
                'label': display,
                'raw_class': pred,
                'score': float(probs[index]),
                'score_percent': round(float(probs[index]) * 100, 1),
                'subscores': subscores,
                'score_label': f'Probabilité estimée ({round(float(probs[index]) * 100, 1)}%)',
                'threshold': None,
                'stage1_tested_percent': round(p_tested * 100, 1),
                'stage2_anormal_percent': round(q_anormal * 100, 1),
                'clinical_action': clinical_action,
                'explanation': explanation,
                'factors': explanation.get('factors', []),
                'priority': explanation.get('priority', {}),
                'actions': explanation.get('actions', []),
                'pharmaco_alerts': explanation.get('pharmaco_alerts', []),
                'patient_context': patient_context,
                'detail': detail
            }
        else:
            features = artifact['features']; row = {}
            for f in features:
                if f in {'number_outpatient', 'number_emergency', 'number_inpatient'}: row[f] = self._numeric(payload, f)
                else: row[f] = self._cat(payload, f)
            X = pd.DataFrame([row], columns=features)
            pipeline = artifact['pipeline']; probs = pipeline.predict_proba(X)[0]; classes = list(pipeline.classes_)
            index = int(np.argmax(probs)); pred = str(classes[index])
            class_labels = artifact.get('class_labels', {})
            display = class_labels.get(pred, pred)
            subscores = [{'label': class_labels.get(str(c), str(c)), 'value': round(float(p) * 100, 1)} for c, p in sorted(zip(classes, probs), key=lambda z: z[1], reverse=True)]
            return {'kind': 'multiclass', 'label': display, 'raw_class': pred, 'score': float(probs[index]), 'score_percent': round(float(probs[index]) * 100, 1),
                    'subscores': subscores, 'score_label': 'Score de classe', 'threshold': None, 'detail': 'Confirmez dans le dossier biologique si le test a réellement été réalisé.'}

    def _predict_treatment(self,payload):
        artifact=self.treatment; features=artifact['features']; row={}
        numeric={'num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses'}
        for f in features:
            row[f]=self._numeric(payload,f) if f in numeric else self._cat(payload,f)
        X=pd.DataFrame([row],columns=features)
        pipe=artifact['pipeline']; proba=pipe.predict_proba(X)[0]; classes=list(pipe.classes_)
        score=float(proba[classes.index(1)]); positive=score>=float(artifact.get('threshold',0.5))
        return {'kind':'binary','label':artifact['positive_label'] if positive else artifact['negative_label'],
                'score':score,'score_percent':round(score*100,1),'threshold':float(artifact.get('threshold',0.5)),
                'positive':positive,'score_label':'Score d’association historique','threshold_label':'Seuil de démonstration 0,50',
                'detail':'Cette estimation décrit une association observée dans un dataset historique. Elle ne doit jamais déclencher seule une prescription ni une modification thérapeutique.'}


def model_manifest():
    p=MODEL_DIR/'manifest.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Load model artifacts once per worker instead of on every submitted form."""
    return ModelService()
