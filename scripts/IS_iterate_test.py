import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, make_scorer
from scipy.io import arff

# 데이터 디렉토리
data_dir = '../data/testarff/'
results_file = 'classification_results.txt'

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

# 데이터 파일 리스트
arff_files = [f for f in os.listdir(data_dir) if f.endswith('.arff')]

# 결과 저장용 딕셔너리
all_results = {}

# 결과 파일 초기화
with open(results_file, 'w') as file:
    file.write("Classification Results\n\n")

# 각 .arff 파일에 대해 반복
for arff_file in arff_files:
    print(f'Processing {arff_file}...')
    
    # 데이터 로드
    data, meta = arff.loadarff(os.path.join(data_dir, arff_file))
    df = pd.DataFrame(data)
    
    # 클래스 열 전처리 (바이너리 문자열 제거)
    df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))

    # 특징 행렬과 라벨 분리
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 레이블 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 파일별 결과 저장용 딕셔너리
    results = {}

    # 각 오버샘플링 기법에 대해 RF와 XGBoost 실행
    for name, sampler in samplers.items():
        print(f'Applying {name}...')
        X_res, y_res = apply_oversampling(sampler, X, y_encoded)

        # K-Fold 설정
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Random Forest 모델 생성 및 학습
        rf = RandomForestClassifier(random_state=42)
        y_pred_rf = cross_val_predict(rf, X_res, y_res, cv=skf, method='predict')
        y_prob_rf = cross_val_predict(rf, X_res, y_res, cv=skf, method='predict_proba')[:, 1]
        report_rf = classification_report(y_res, y_pred_rf, target_names=le.classes_)
        roc_auc_rf = roc_auc_score(y_res, y_prob_rf)
        precision_rf, recall_rf, _ = precision_recall_curve(y_res, y_prob_rf)
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
        # xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        # y_pred_xgb = cross_val_predict(xgb, X_res, y_res, cv=skf, method='predict')
        # y_prob_xgb = cross_val_predict(xgb, X_res, y_res, cv=skf, method='predict_proba')[:, 1]
        # report_xgb = classification_report(y_res, y_pred_xgb, target_names=le.classes_)
        # roc_auc_xgb = roc_auc_score(y_res, y_prob_xgb)
        # precision_xgb, recall_xgb, _ = precision_recall_curve(y_res, y_prob_xgb)
        # pr_auc_xgb = auc(recall_xgb, precision_xgb)
        # results[f'{name}_XGB'] = {
        #     'classification_report': report_xgb,
        #     'roc_auc': roc_auc_xgb,
        #     'pr_auc': pr_auc_xgb
        # }
        # print(f'{name} XGB classification report:\n', report_xgb)
        # print(f'{name} XGB ROC-AUC: {roc_auc_xgb}')
        # print(f'{name} XGB PR-AUC: {pr_auc_xgb}')
    
    # 각 파일의 결과를 전체 결과에 저장
    all_results[arff_file] = results

    # 결과를 텍스트 파일에 저장
    with open(results_file, 'a') as file:
        file.write(f'Results for {arff_file}:\n')
        for name, result in results.items():
            file.write(f'{name} classification report:\n{result["classification_report"]}\n')
            file.write(f'{name} ROC-AUC: {result["roc_auc"]}\n')
            file.write(f'{name} PR-AUC: {result["pr_auc"]}\n\n')

# 전체 결과 출력 (선택 사항)
for arff_file, metrics in all_results.items():
    print(f'Results for {arff_file}:')
    for name, result in metrics.items():
        print(f'{name} classification report:\n{result["classification_report"]}\n')
        print(f'{name} ROC-AUC: {result["roc_auc"]}')
        print(f'{name} PR-AUC: {result["pr_auc"]}\n')
