import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN
import pickle

# 데이터 로드 및 전처리
file_path = '../data/arff/glass-0-1-2-3_vs_4-5-6.dat_cleaned.arff'
data = pd.read_csv(file_path, comment='@', header=None)
data.columns = [f'feature_{i}' for i in range(data.shape[1]-1)] + ['class']

# 레이블 인코딩
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

# Positive class 데이터만 추출
positive_data = data[data['class'] == 1].drop(columns=['class'])
from ctgan import CTGAN
# CTGAN 모델 설정 및 학습

ctgan = CTGAN()

# 에포크 수 설정
epochs = 100

model = ctgan
model.fit(positive_data, epochs=epochs)

# 데이터 생성
generated_data = model.sample(100)
generated_data['class'] = 1

# 생성된 데이터셋을 pickle로 저장
with open('generated_data.pkl', 'wb') as f:
    pickle.dump(generated_data, f)

# 기존 데이터와 생성된 데이터 결합
combined_data = pd.concat([data, generated_data], ignore_index=True)

# 특징과 레이블 분할
X_original = data.drop(columns=['class'])
y_original = data['class']
X_combined = combined_data.drop(columns=['class'])
y_combined = combined_data['class']

# 데이터 분할 (원본 데이터셋)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

# 데이터 분할 (결합된 데이터셋)
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# Random Forest 모델 생성 및 학습 (원본 데이터셋)
rf_orig = RandomForestClassifier(random_state=42)
rf_orig.fit(X_train_orig, y_train_orig)
y_pred_rf_orig = rf_orig.predict(X_test_orig)
y_prob_rf_orig = rf_orig.predict_proba(X_test_orig)[:, 1]

# Random Forest 모델 생성 및 학습 (결합된 데이터셋)
rf_comb = RandomForestClassifier(random_state=42)
rf_comb.fit(X_train_comb, y_train_comb)
y_pred_rf_comb = rf_comb.predict(X_test_comb)
y_prob_rf_comb = rf_comb.predict_proba(X_test_comb)[:, 1]

# 평가 지표 계산 (원본 데이터셋)
report_rf_orig = classification_report(y_test_orig, y_pred_rf_orig, target_names=le.classes_)
roc_auc_rf_orig = roc_auc_score(y_test_orig, y_prob_rf_orig)
precision_rf_orig, recall_rf_orig, _ = precision_recall_curve(y_test_orig, y_prob_rf_orig)
pr_auc_rf_orig = auc(recall_rf_orig, precision_rf_orig)

# 평가 지표 계산 (결합된 데이터셋)
report_rf_comb = classification_report(y_test_comb, y_pred_rf_comb, target_names=le.classes_)
roc_auc_rf_comb = roc_auc_score(y_test_comb, y_prob_rf_comb)
precision_rf_comb, recall_rf_comb, _ = precision_recall_curve(y_test_comb, y_prob_rf_comb)
pr_auc_rf_comb = auc(recall_rf_comb, precision_rf_comb)

# 결과 출력
print('Original Data RF classification report:\n', report_rf_orig)
print('Original Data RF ROC-AUC:', roc_auc_rf_orig)
print('Original Data RF PR-AUC:', pr_auc_rf_orig)

print('\nCombined Data RF classification report:\n', report_rf_comb)
print('Combined Data RF ROC-AUC:', roc_auc_rf_comb)
print('Combined Data RF PR-AUC:', pr_auc_rf_comb)
