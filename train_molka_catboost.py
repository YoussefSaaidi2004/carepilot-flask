import sys, re, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

from catboost_model import CatWrap, TwoStageClassifier, group_diag, TE_COLS

def main():
    csv_path = Path('data/diabetic_data.csv')
    print('Loading dataset from', csv_path)
    df = pd.read_csv(csv_path, keep_default_na=False)

    # 1. Deduplication by patient_nbr
    df = df.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first').copy()
    print('Shape after deduplication:', df.shape)

    # 2. Target merge
    def merge_target(val):
        if val in ['>7', '>8']:
            return 'Anormal'
        return val

    df['A1C_target'] = df['A1Cresult'].apply(merge_target)
    y = df['A1C_target']

    # 3. Drops and cleaning
    cols_to_drop = ['weight', 'payer_code', 'encounter_id']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    for col in ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].replace({'?': 'Missing', '': 'Missing'}).fillna('Missing')

    race_mode = df['race'].replace({'?': np.nan}).mode()[0]
    df['race'] = df['race'].replace({'?': race_mode})

    # Drop zero variance
    near_constant = [col for col in df.columns if df[col].nunique() <= 1]
    if near_constant:
        df = df.drop(columns=near_constant)

    # Raw categories for CatBoost
    raw_cat = pd.DataFrame({
        'diag_1_3': df['diag_1'].astype(str).str[:3],
        'diag_2_3': df['diag_2'].astype(str).str[:3],
        'diag_3_3': df['diag_3'].astype(str).str[:3],
        'specialty_cat': df['medical_specialty'].astype(str),
        'discharge_cat': df['discharge_disposition_id'].astype(str),
    }, index=df.index)

    # Prepare features
    X_raw = df.drop(columns=['A1Cresult', 'A1C_target', 'patient_nbr'])

    # Diagnostics grouping
    for col in ['diag_1', 'diag_2', 'diag_3']:
        X_raw[col + '_group'] = X_raw[col].apply(group_diag)
    X_raw = X_raw.drop(columns=['diag_1', 'diag_2', 'diag_3'])

    # Age ordinal encoding
    age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                 '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
    age_map = {cat: i for i, cat in enumerate(age_order)}
    X_raw['age_encoded'] = X_raw['age'].map(lambda a: age_map.get(a, 5))
    X_raw = X_raw.drop(columns=['age'])

    # Categorical nominal
    categorical_nominal = ['race', 'gender', 'medical_specialty',
                           'diag_1_group', 'diag_2_group', 'diag_3_group',
                           'max_glu_serum', 'change', 'diabetesMed']
    non_numeric_cols = X_raw.select_dtypes(exclude='number').columns.tolist()
    med_cols = [c for c in non_numeric_cols if c not in categorical_nominal]
    categorical_nominal += med_cols

    X_encoded = pd.get_dummies(X_raw, columns=categorical_nominal, drop_first=True)

    # Numerical scaling
    num_cols = ['num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient',
                'number_diagnoses', 'time_in_hospital']
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols].astype(float))

    # Interaction & clinical features
    total_visits = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    med_cols_raw = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
                    'glipizide','glyburide','pioglitazone','rosiglitazone','insulin']
    present_meds = [m for m in med_cols_raw if m in df.columns]
    nb_meds_ch = (df[present_meds] != 'No').sum(axis=1)

    visit_scaler = StandardScaler()
    med_ch_scaler = StandardScaler()
    X_encoded['total_visits_prior'] = visit_scaler.fit_transform(total_visits.to_frame())
    X_encoded['nb_meds_changed'] = med_ch_scaler.fit_transform(nb_meds_ch.to_frame())

    days = df['time_in_hospital'].clip(lower=1)
    grp = df[['diag_1', 'diag_2', 'diag_3']].apply(lambda col: col.map(group_diag))

    X_encoded['meds_per_day'] = df['num_medications'] / days
    X_encoded['labs_per_day'] = df['num_lab_procedures'] / days
    X_encoded['diag_per_day'] = df['number_diagnoses'] / days
    X_encoded['nb_meds_dose_changed'] = df[present_meds].isin(['Up', 'Down']).sum(axis=1)
    X_encoded['insulin_dose_changed'] = df['insulin'].isin(['Up', 'Down']).astype(int) if 'insulin' in df.columns else 0
    X_encoded['nb_diag_diabetes'] = (grp == 'Diabete').sum(axis=1)
    X_encoded['nb_diag_groups'] = grp.nunique(axis=1)
    X_encoded['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
    X_encoded['high_utilizer'] = (total_visits >= 3).astype(int)
    X_encoded['emergency_admission'] = (df['admission_type_id'].astype(str) == '1').astype(int)
    X_encoded['age_senior'] = (X_encoded['age_encoded'] >= 7).astype(int)
    X_encoded['polypharmacy'] = (df['num_medications'] >= 15).astype(int)

    # Normalize column names
    clean_cols = [re.sub(r'[^0-9a-zA-Z_]+', '_', c) for c in X_encoded.columns]
    X_encoded.columns = clean_cols

    # Join raw_cat for CatBoost
    X_final = X_encoded.join(raw_cat)
    feature_names = list(X_final.columns)
    print('Final feature matrix shape:', X_final.shape)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    print('Training TwoStageClassifier with CatBoost (300 iterations per stage)...')
    model = TwoStageClassifier(stage1=CatWrap(iterations=300), stage2=CatWrap(iterations=300))
    model.fit(X_train, y_train)

    # Evaluation on test
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    f1_m = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(pd.get_dummies(y_test), probs, multi_class='ovr')

    print(f'=== TEST RESULTS (Molka Jebali TwoStage CatBoost) ===')
    print(f'F1 macro : {f1_m:.3f}')
    print(f'Accuracy : {acc:.3f}')
    print(f'ROC-AUC  : {roc:.3f}')
    print(classification_report(y_test, preds, digits=3))

    # Medians and modes for inference default filling
    medians = {col: float(df[col].median()) for col in num_cols if col in df.columns}
    medians['number_diagnoses'] = float(df['number_diagnoses'].median())

    modes = {
        'race': race_mode,
        'gender': 'Female',
        'medical_specialty': 'Missing',
        'admission_type_id': '1',
        'admission_source_id': '7',
        'discharge_disposition_id': '1',
        'change': 'No',
        'diabetesMed': 'Yes',
        'max_glu_serum': 'None',
        'insulin': 'No'
    }

    # Save artifact
    out_path = Path('models/hba1c_status.joblib')
    artifact = {
        'model': model,
        'model_name': 'Modèle séquentiel à deux étages',
        'classes': list(model.classes_), # ['Anormal', 'None', 'Norm']
        'feature_names': feature_names,
        'dummy_columns': list(X_encoded.columns),
        'scaler': scaler,
        'num_cols': num_cols,
        'visit_scaler': visit_scaler,
        'med_ch_scaler': med_ch_scaler,
        'age_map': age_map,
        'categorical_nominal': categorical_nominal,
        'medians': medians,
        'modes': modes,
        'f1_macro': round(f1_m, 3),
        'accuracy': round(acc * 100, 1),
        'roc_auc': round(roc, 3),
        'class_labels': {
            'None': 'Absence probable de test documenté',
            'Norm': 'Résultat HbA1c normalisé (< 7%)',
            'Anormal': 'Résultat HbA1c anormal (> 7% ou > 8%)'
        }
    }
    joblib.dump(artifact, out_path, compress=3)
    print('Saved CatBoost artifact to', out_path, f'({out_path.stat().st_size / 1e6:.2f} MB)')

if __name__ == '__main__':
    main()
