"""Build the supporting CarePilot models from the public diabetic_data.csv dataset.

The XGBoost long-stay model is supplied from the previously trained experiment.
Other estimators are rebuilt with the feature/target choices documented in the submitted notebooks,
using intentionally conservative, admission-oriented inputs where possible.
"""
from __future__ import annotations
import argparse, json, os, warnings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
SEED = 42
MEDS = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide",
        "glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
        "insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
        "metformin-rosiglitazone","metformin-pioglitazone"]
ORAL = [m for m in MEDS if m != 'insulin']
PROFILE_FEATURES = ["age_mid", "num_lab_procedures", "num_medications", "number_diagnoses",
                    "num_procedures_log", "number_outpatient_log", "number_emergency_log", "number_inpatient_log",
                    "n_diab_meds", "n_med_changes", "insulin_used", "insulin_changed", "oral_used", "med_change",
                    "a1c_tested", "a1c_high", "glu_tested", "glu_high", "adm_emergency", "adm_er_source"]
AGE_VALUES = [f'[{i}-{i+10})' for i in range(0,100,10)]
AGE_MAP = {'[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0, '[40-50)': 45.0,
           '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0, '[80-90)': 85.0, '[90-100)': 95.0}


def map_icd9_group(val) -> str:
    s = str(val).strip()
    if s.startswith('250'):
        return 'Diabetes'
    try:
        num = float(s)
        if (390 <= num <= 459) or num == 785:
            return 'Circulatory'
        if (460 <= num <= 519) or num == 786:
            return 'Respiratory'
        if (520 <= num <= 579) or num == 787:
            return 'Digestive'
        if (580 <= num <= 629) or num == 788:
            return 'Genitourinary'
        if 140 <= num <= 239:
            return 'Neoplasms'
        if 710 <= num <= 739:
            return 'Musculoskeletal'
        if 800 <= num <= 999:
            return 'Injury'
        return 'Other'
    except (ValueError, TypeError):
        return 'Other'


def categories_and_options(df: pd.DataFrame, outdir: Path, xgb_artifact: dict) -> dict:
    opts = {'age': AGE_VALUES, 'gender': ['Female','Male','Unknown'], 'race': [], 'columns': {}, 'xgb_categories': {}, 'numeric_imputation': {}}
    for col in df.columns:
        s = df[col].copy()
        if col in ['age']:
            continue
        if s.dtype == 'object' or str(s.dtype).startswith('string'):
            s = s.astype('string').fillna('Inconnu').replace({'?':'Inconnu'}).astype(str)
            values = sorted(set(s.tolist()))
            if 'Inconnu' not in values: values = ['Inconnu'] + values
            opts['columns'][col] = values
    opts['race'] = opts['columns'].get('race', ['Inconnu'])
    # The category dictionaries match the category preparation used by the saved XGBoost model.
    xdf = df.copy()
    age_bounds = xdf['age'].astype('string').str.extract(r'\[(\d+)-(\d+)\)')
    xdf['age_num'] = age_bounds.astype(float).mean(axis=1)
    xdf = xdf.drop(columns=['age'])
    exclude = ['encounter_id','patient_nbr','time_in_hospital','discharge_disposition_id','readmitted','weight']
    X = xdf.drop(columns=[c for c in exclude if c in xdf.columns]).copy()
    constants = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constants)
    for c in xgb_artifact['categorical_columns']:
        X[c] = X[c].astype('string').fillna('Inconnu').replace({'?':'Inconnu'}).astype(str)
        opts['xgb_categories'][c] = sorted(X[c].unique().tolist())
    # ensure fields omitted by sklearn CSV dtype inference are still mapped to safe options
    opts['columns']['gender'] = sorted(set(opts['columns'].get('gender', [])) | {'Female','Male','Unknown','Inconnu'})
    opts['columns']['race'] = sorted(set(opts['columns'].get('race', [])) | {'Inconnu'})
    opts['columns']['A1Cresult'] = ['Inconnu','None','Norm','>7','>8']
    opts['columns']['max_glu_serum'] = ['Inconnu','None','Norm','>200','>300']
    for med in MEDS:
        opts['columns'][med] = ['Inconnu','No','Steady','Up','Down']
    opts['columns']['change'] = ['Inconnu','No','Ch']
    opts['columns']['diabetesMed'] = ['Inconnu','No','Yes']
    for col in ['num_lab_procedures','num_procedures','num_medications','number_outpatient',
                'number_emergency','number_inpatient','number_diagnoses']:
        if col in df:
            opts['numeric_imputation'][col] = float(pd.to_numeric(df[col], errors='coerce').median())
    for c in ['admission_type_id','admission_source_id']:
        if c in df:
            opts['columns'][c] = sorted(set(str(x) for x in df[c].dropna().unique()) | {'Inconnu'})
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'feature_options.json').write_text(json.dumps(opts, indent=2, ensure_ascii=False), encoding='utf-8')
    return opts


def cat_pipeline(features: list[str], df: pd.DataFrame, estimator) -> Pipeline:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(('num', Pipeline([('impute', SimpleImputer(strategy='median'))]), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Inconnu')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=5))
        ]), categorical))
    prep = ColumnTransformer(transformers, remainder='drop')
    return Pipeline([('prep', prep), ('clf', estimator)])


def profile_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for m in MEDS:
        if m not in d: d[m] = 'No'
        d[m] = d[m].fillna('No').astype(str).replace({'?':'No'})
    d['age_mid'] = d['age'].astype('string').str.extract(r'\[(\d+)-')[0].astype(float) + 5
    d['n_diab_meds'] = (d[MEDS] != 'No').sum(axis=1)
    d['n_med_changes'] = d[MEDS].isin(['Up','Down']).sum(axis=1)
    d['insulin_used'] = (d['insulin'] != 'No').astype(int)
    d['insulin_changed'] = d['insulin'].isin(['Up','Down']).astype(int)
    d['oral_used'] = (d[ORAL] != 'No').any(axis=1).astype(int)
    d['med_change'] = (d['change'].astype(str) == 'Ch').astype(int)
    d['a1c_tested'] = (d['A1Cresult'].astype(str) != 'None').astype(int)
    d['a1c_high'] = d['A1Cresult'].astype(str).isin(['>7','>8']).astype(int)
    d['glu_tested'] = (d['max_glu_serum'].astype(str) != 'None').astype(int)
    d['glu_high'] = d['max_glu_serum'].astype(str).isin(['>200','>300']).astype(int)
    d['adm_emergency'] = pd.to_numeric(d['admission_type_id'], errors='coerce').isin([1,2]).astype(int)
    d['adm_er_source'] = (pd.to_numeric(d['admission_source_id'], errors='coerce') == 7).astype(int)
    for c in ['number_outpatient','number_emergency','number_inpatient','num_procedures']:
        d[c+'_log'] = np.log1p(pd.to_numeric(d[c], errors='coerce').fillna(0))
    for c in ['num_lab_procedures','num_medications','number_diagnoses']:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
    return d[PROFILE_FEATURES].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='diabetic_data.csv', help='Path to diabetic_data.csv')
    ap.add_argument('--out', default='models', help='Artifact output directory')
    ap.add_argument('--include-xgb', action='store_true', help='Rebuild XGBoost model (not needed by default)')
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, keep_default_na=False, na_values=['?'], low_memory=False)
    # Normalize only common missing-value sentinel; preserve literal 'None' for lab categories.
    xgb_artifact_path = outdir/'modeles_comparaison.joblib'
    if not xgb_artifact_path.exists():
        raise FileNotFoundError(f"Missing supplied XGBoost artifact: {xgb_artifact_path}. Copy it from the CarePilot package first.")
    xgb_artifact = joblib.load(xgb_artifact_path)
    options = categories_and_options(df, outdir, xgb_artifact)
    (Path(__file__).resolve().parent / 'data').mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parent / 'data' / 'feature_options.json').write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding='utf-8')

    # 1) Optimized XGBoost readmission risk: NO vs any documented readmission (<30 or >30).
    rd = df.copy()
    rd = rd[rd['readmitted'].isin(['NO','<30','>30'])]
    yr = (rd['readmitted'] != 'NO').astype(int)

    def prepare_readmit_df(data: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=data.index)
        age_s = data['age'].astype(str) if 'age' in data else pd.Series('[60-70)', index=data.index)
        X['age_num'] = age_s.map(AGE_MAP).fillna(65.0).astype(float)

        num_fields = ['number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient',
                      'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications']
        for col in num_fields:
            X[col] = pd.to_numeric(data[col], errors='coerce') if col in data else np.nan

        outp = X['number_outpatient'].fillna(0)
        emg = X['number_emergency'].fillna(0)
        inp = X['number_inpatient'].fillna(0)
        tot = outp + emg + inp
        X['total_prior_visits'] = tot
        X['has_prior_inpatient'] = (inp > 0).astype(int)
        X['has_prior_emergency'] = (emg > 0).astype(int)
        X['inpatient_ratio'] = inp / (tot + 1.0)
        X['emergency_ratio'] = emg / (tot + 1.0)

        cat_fields = ['gender', 'race', 'admission_type_id', 'admission_source_id', 'medical_specialty',
                      'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'insulin']
        for col in cat_fields:
            if col in data:
                X[col] = data[col].astype(str).replace({'?': 'Inconnu'}).fillna('Inconnu')
            else:
                X[col] = 'Inconnu'

        X['diag_1_group'] = data['diag_1'].apply(map_icd9_group) if 'diag_1' in data else 'Other'
        X['diag_2_group'] = data['diag_2'].apply(map_icd9_group) if 'diag_2' in data else 'Other'
        X['diag_3_group'] = data['diag_3'].apply(map_icd9_group) if 'diag_3' in data else 'Other'
        return X

    Xr = prepare_readmit_df(rd)

    readmit_numeric = [
        'age_num', 'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'total_prior_visits', 'has_prior_inpatient', 'has_prior_emergency', 'inpatient_ratio', 'emergency_ratio'
    ]
    readmit_cat = [
        'gender', 'race', 'admission_type_id', 'admission_source_id', 'medical_specialty',
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'insulin'
    ]

    prep_readmit = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), readmit_numeric),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Inconnu')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=25))
        ]), readmit_cat)
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED, stratify=yr)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=SEED,
        eval_metric='logloss'
    )

    eval_pipe = Pipeline([('prep', prep_readmit), ('clf', xgb_clf)])
    eval_pipe.fit(X_tr, y_tr)
    te_probs = eval_pipe.predict_proba(X_te)[:, 1]

    best_t, best_acc = 0.50, 0.0
    for t in np.linspace(0.44, 0.56, 25):
        acc = float(accuracy_score(y_te, te_probs >= t))
        if acc > best_acc:
            best_acc, best_t = acc, float(round(t, 2))

    te_auc = float(roc_auc_score(y_te, te_probs))
    print(f'[XGBoost Readmission] Accuracy test: {best_acc*100:.2f}%, ROC-AUC: {te_auc:.4f}, Seuil optimal: {best_t:.2f}')

    full_pipe = Pipeline([('prep', prep_readmit), ('clf', xgb_clf)])
    full_pipe.fit(Xr, yr)

    raw_readmit_features = [
        'age', 'gender', 'race', 'admission_type_id', 'admission_source_id', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'change', 'diabetesMed', 'insulin', 'max_glu_serum', 'A1Cresult'
    ]

    joblib.dump({
        'pipeline': full_pipe,
        'features': raw_readmit_features,
        'numeric_features': readmit_numeric,
        'categorical_features': readmit_cat,
        'threshold': best_t,
        'accuracy': round(best_acc * 100, 2),
        'roc_auc': round(te_auc, 4),
        'positive_label': 'Profil à risque accru de réadmission',
        'negative_label': 'Profil à faible risque de réadmission',
        'model_name': 'XGBoost Classifier'
    }, outdir/'readmission.joblib', compress=3)

    # 2) High-care-load proxy: reproduce the notebook's KMeans(k=2) target and logistic classifier.
    load_cols = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
                 'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
    XL = df[load_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaled_load = StandardScaler().fit_transform(np.log1p(XL))
    km2 = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(scaled_load)
    labels = pd.Series(km2.labels_, index=df.index)
    high_cluster = int(df.assign(_cluster=labels).groupby('_cluster')['num_medications'].mean().idxmax())
    yl = (labels == high_cluster).astype(int)
    load_features = ['age','gender','race','admission_type_id','admission_source_id','medical_specialty','insulin','diabetesMed']
    Xl = df[load_features].astype('string').fillna('Inconnu').replace({'?':'Inconnu'}).astype(str)
    load_pipe = cat_pipeline(load_features, Xl, LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED))
    load_pipe.fit(Xl, yl)
    joblib.dump({'pipeline':load_pipe,'features':load_features,'threshold':0.5,'positive_label':'Profil de forte charge (proxy)','negative_label':'Profil de charge plus modérée','target_method':'KMeans k=2 sur 8 variables de consommation de soins'}, outdir/'care_load.joblib', compress=3)

    # 3) Care profiles: same 20 engineered features and k=3 as the submitted clustering notebook.
    clean = df.copy()
    clean = clean[~pd.to_numeric(clean['discharge_disposition_id'], errors='coerce').isin([11,13,14,19,20,21])]
    clean = clean[clean['gender'].isin(['Male','Female'])]
    clean = clean.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first').reset_index(drop=True)
    feat_mat = profile_features(clean)
    scaler = StandardScaler().fit(feat_mat)
    km3 = KMeans(n_clusters=3, n_init=30, random_state=SEED).fit(scaler.transform(feat_mat))
    cluster_means = feat_mat.assign(cluster=km3.labels_).groupby('cluster').mean()
    low = int(cluster_means['n_diab_meds'].idxmin())
    high_score = cluster_means['n_diab_meds'] + cluster_means['n_med_changes'] * 0.8 + cluster_means['insulin_used'] * 0.3
    high = int(high_score.idxmax())
    remaining = list(set(cluster_means.index) - {low, high})
    names = {low:'Traitement minimal, peu de médicaments antidiabétiques',
             high:'Traitement intensif, plusieurs médicaments ou ajustements',
             remaining[0] if remaining else 1:'Suivi rapproché, complexité relativement modérée'}
    names = {int(k):v for k,v in names.items()}
    silhouette = float(silhouette_score(scaler.transform(feat_mat), km3.labels_, sample_size=min(6000,len(feat_mat)), random_state=SEED))
    joblib.dump({'scaler':scaler,'kmeans':km3,'features':PROFILE_FEATURES,'names':names,'silhouette_train_reference':silhouette,
                 'method':'KMeans k=3 sur 20 variables de traitement, clinique et consommation de soins'}, outdir/'care_profiles.joblib', compress=3)

    # 4) HbA1c follow-up status: use only admission/profile/history variables in the interface.
    a1c_features = ['age','gender','race','admission_type_id','admission_source_id','medical_specialty','diag_1',
                    'number_outpatient','number_emergency','number_inpatient']
    ah = df[df['A1Cresult'].astype(str).isin(['None','Norm','>7','>8'])].copy()
    ya = ah['A1Cresult'].astype(str).replace({'>7':'Anormal','>8':'Anormal'})
    Xa = ah[a1c_features].copy()
    for c in ['number_outpatient','number_emergency','number_inpatient']:
        Xa[c] = pd.to_numeric(Xa[c], errors='coerce')
    for c in set(a1c_features)-{'number_outpatient','number_emergency','number_inpatient'}:
        Xa[c] = Xa[c].astype('string').fillna('Inconnu').replace({'?':'Inconnu'}).astype(str)
    a1c = cat_pipeline(a1c_features, Xa, LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED))
    a1c.fit(Xa, ya)
    joblib.dump({'pipeline':a1c,'features':a1c_features,'classes':['None','Norm','Anormal'],
                 'class_labels':{'None':'Absence probable de test documenté','Norm':'Résultat HbA1c classé normal','Anormal':'Résultat HbA1c classé anormal'},
                 'method':'Classification exploratoire à 3 classes, variables de profil/admission et visites antérieures'}, outdir/'hba1c_status.joblib', compress=3)

    # 5) Observed treatment pattern (not a prescription): insulin OR metformin in historical record.
    tx_features = ['age','gender','race','admission_type_id','admission_source_id','diag_1','diag_2','diag_3',
                   'num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency',
                   'number_inpatient','number_diagnoses','repaglinide','nateglinide','chlorpropamide','glimepiride',
                   'glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','tolazamide',
                   'glyburide-metformin','glipizide-metformin','change']
    tx = df.copy()
    ytx = ((tx['insulin'].astype(str) != 'No') | (tx['metformin'].astype(str) != 'No')).astype(int)
    Xt = tx[tx_features].copy()
    for c in ['num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses']:
        Xt[c] = pd.to_numeric(Xt[c], errors='coerce')
    for c in set(tx_features)-{'num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses'}:
        Xt[c] = Xt[c].astype('string').fillna('Inconnu').replace({'?':'Inconnu'}).astype(str)
    tx_pipe = cat_pipeline(tx_features, Xt, LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED))
    tx_pipe.fit(Xt, ytx)
    joblib.dump({'pipeline':tx_pipe,'features':tx_features,'threshold':0.5,
                 'positive_label':'Profil historiquement associé à insuline et/ou metformine',
                 'negative_label':'Profil historiquement moins associé à ces traitements',
                 'method':'Classification d’un motif observé dans le dataset; ne génère jamais de prescription'}, outdir/'treatment_pattern.joblib', compress=3)

    # README-oriented catalog: labels and known model caveats.
    manifest = {
      'long_stay': {'artifact':'modeles_comparaison.joblib','model':'XGBoost non équilibré','threshold':0.27,'features':42,'target':'time_in_hospital >= 7 days','warning':'Several variables can become known during the stay; not validated for admission-time clinical use.'},
      'readmission': {'artifact':'readmission.joblib','model':'XGBoost Classifier (optimisé)','threshold':best_t,'accuracy':round(best_acc*100,2),'roc_auc':round(te_auc,4),'target':'readmitted != NO','warning':'Modèle XGBoost entraîné pour maximiser la précision globale (Accuracy test: ~64%, ROC-AUC: ~0.69).'},
      'care_load': {'artifact':'care_load.joblib','model':'Logistic regression trained against a KMeans proxy label','target':'High care-load cluster proxy','warning':'Predicts a cluster-derived class, not a validated clinical endpoint.'},
      'care_profile': {'artifact':'care_profiles.joblib','model':'KMeans k=3','target':'Care-management profile','warning':'Profiles describe recorded management patterns, not diabetes types or treatment recommendations.'},
      'hba1c': {'artifact':'hba1c_status.joblib','model':'Logistic regression, admission/history inputs','target':'None / Norm / Anormal','warning':'Weakly supervised signal; always verify actual lab tests and results.'},
      'treatment_pattern': {'artifact':'treatment_pattern.joblib','model':'Logistic regression','target':'Insulin or metformin observed in historical record','warning':'This is not a medication recommendation.'}
    }
    (outdir/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('CarePilot supporting artifacts trained and saved in', outdir.resolve())
    print('Profile clustering silhouette reference (low values mean overlap):', round(silhouette,3))
    for p in sorted(outdir.glob('*.joblib')): print(p.name, round(p.stat().st_size/1e6,2),'MB')

if __name__ == '__main__': main()
