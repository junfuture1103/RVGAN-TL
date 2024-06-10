import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

# 데이터 로드
from scipy.io import arff
data, meta = arff.loadarff('../data/arff/yeast1.dat_cleaned.arff')
df = pd.DataFrame(data)

# 클래스 열 전처리 (바이너리 문자열 제거)
df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))

# 특징 행렬과 라벨 분리
X = df.drop('Class', axis=1)
y = df['Class']

# 레이블 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 오버샘플링 기법 적용 함수
def apply_oversampling(sampler, X, y):
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

# 오버샘플링 기법들
samplers = {
    'ROS': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
}

# 결과 저장용 딕셔너리
results = {}

# 각 오버샘플링 기법에 대해 RF와 XGBoost 실행
for name, sampler in samplers.items():
    print(f'Applying {name}...')
    X_res, y_res = apply_oversampling(sampler, X, y_encoded)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    
    # Random Forest 모델 생성 및 학습
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    report_rf = classification_report(y_test, y_pred_rf, target_names=le.classes_)
    roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    pr_auc_rf = auc(recall_rf, precision_rf)
    results[f'{name}_RF'] = {
        'classification_report': report_rf,
        'roc_auc': roc_auc_rf,
        'pr_auc': pr_auc_rf
    }
    print(f'{name} RF classification report:\n', report_rf)
    print(f'{name} RF ROC-AUC: {roc_auc_rf}')
    print(f'{name} RF PR-AUC: {pr_auc_rf}')
    
    # XGBoost 모델 생성 및 학습
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    report_xgb = classification_report(y_test, y_pred_xgb, target_names=le.classes_)
    roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    pr_auc_xgb = auc(recall_xgb, precision_xgb)
    results[f'{name}_XGB'] = {
        'classification_report': report_xgb,
        'roc_auc': roc_auc_xgb,
        'pr_auc': pr_auc_xgb
    }
    print(f'{name} XGB classification report:\n', report_xgb)
    print(f'{name} XGB ROC-AUC: {roc_auc_xgb}')
    print(f'{name} XGB PR-AUC: {pr_auc_xgb}')

# 결과 출력
for name, metrics in results.items():
    print(f'{name} classification report:\n{metrics["classification_report"]}\n')
    print(f'{name} ROC-AUC: {metrics["roc_auc"]}')
    print(f'{name} PR-AUC: {metrics["pr_auc"]}\n')
