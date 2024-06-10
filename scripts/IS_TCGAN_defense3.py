import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from ctgan import CTGAN
import torch
import torch.nn as nn
from torch.autograd import Variable

# 데이터 로드
file_path = '../data/testarff/yeast-0-2-5-6_vs_3-7-8-9.dat_cleaned.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# 클래스 열 전처리 (바이너리 문자열 제거)
df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))

# 특징 행렬과 라벨 분리
X = df.drop('Class', axis=1)
y = df['Class']

# 레이블 인코딩 
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 모델 성능 평가
y_pred = model.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)

# LowProFool 공격
def lowProFool(X, model, weights, bounds, maxiters=100, alpha=0.02, lambda_=0.1):
    X_adv = X.copy()
    r = np.zeros_like(X_adv)
    
    for _ in range(maxiters):
        grad = np.zeros_like(X_adv)
        for i in range(len(X_adv)):
            X_adv[i] += alpha
            y_pred = model.predict_proba(X_adv)
            loss = -np.log(y_pred[np.arange(len(y_pred)), y_test])
            grad[i] = (loss.sum() / len(loss)) / alpha
            X_adv[i] -= alpha
        
        r += alpha * np.sign(grad)
        r = np.clip(r, -bounds, bounds)
        X_adv = X + r
    
    return X_adv

# 적대적 샘플 생성
weights = np.ones(X_test.shape[1])
bounds = 0.1  # Adjust this value as per the dataset
X_test_adv = lowProFool(X_test.values, model, weights, bounds)

# 적대적 샘플에 대한 모델 성능 평가
y_pred_adv = model.predict(X_test_adv)
adv_accuracy = accuracy_score(y_test, y_pred_adv)

# 강건성 평가 메트릭 추가
precision = precision_score(y_test, y_pred_adv, average='macro')
recall = recall_score(y_test, y_pred_adv, average='macro')
f1 = f1_score(y_test, y_pred_adv, average='macro')

print(f'Initial accuracy: {initial_accuracy}')
print(f'Accuracy on adversarial samples: {adv_accuracy}')
print(f'Precision on adversarial samples: {precision}')
print(f'Recall on adversarial samples: {recall}')
print(f'F1 Score on adversarial samples: {f1}')

# 오분류된 샘플 추출
misclassified_indices = np.where(y_test != y_pred_adv)[0]
misclassified_samples = pd.DataFrame(X_test_adv[misclassified_indices], columns=X_test.columns)
misclassified_labels = y_test[misclassified_indices]
misclassified_predictions = y_pred_adv[misclassified_indices]

print("Misclassified adversarial samples:")
print(misclassified_samples)
print("Misclassified labels (true):")
print(misclassified_labels)
print("Misclassified predictions (predicted):")
print(misclassified_predictions)

# TCGAN으로 데이터 생성
data_to_generate = misclassified_samples.copy()
data_to_generate['Class'] = misclassified_labels

ctgan = CTGAN()
ctgan.fit(data_to_generate, ['Class'])

num_samples_to_generate = len(data_to_generate)
generated_samples = ctgan.sample(num_samples_to_generate)

# 새로운 학습 데이터 구성
new_X_train = pd.concat([X_train, generated_samples.drop('Class', axis=1)])
new_y_train = np.concatenate([y_train, generated_samples['Class'].values])

new_X_train = pd.concat([new_X_train, misclassified_samples])
new_y_train = np.concatenate([new_y_train, misclassified_labels])

# 모델 재학습
model.fit(new_X_train, new_y_train)

# 새로운 적대적 샘플에 대한 성능 평가
new_y_pred_adv = model.predict(X_test_adv)
new_adv_accuracy = accuracy_score(y_test, new_y_pred_adv)

new_precision = precision_score(y_test, new_y_pred_adv, average='macro')
new_recall = recall_score(y_test, new_y_pred_adv, average='macro')
new_f1 = f1_score(y_test, new_y_pred_adv, average='macro')

print(f'New accuracy on adversarial samples after retraining: {new_adv_accuracy}')
print(f'New precision on adversarial samples after retraining: {new_precision}')
print(f'New recall on adversarial samples after retraining: {new_recall}')
print(f'New F1 Score on adversarial samples after retraining: {new_f1}')

# 차원 축소 (PCA)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(new_X_train)
X_test_pca = pca.transform(X_test)
X_test_adv_pca = pca.transform(X_test_adv)

# KNN 모델 훈련
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, new_y_train)

# 원본 데이터 예측
y_test_pred = knn.predict(X_test_pca)

# 적대적 샘플 데이터 예측
y_test_adv_pred = knn.predict(X_test_adv_pca)

# 시각화를 위한 데이터 프레임 생성 (라벨 0과 1을 구분하여 시각화)
visualization_df = pd.DataFrame({
    'pca_0': X_test_pca[:, 0],
    'pca_1': X_test_pca[:, 1],
    'label': y_test,
    'prediction': y_test_pred,
    'type': 'original'
})

# 적대적 샘플을 데이터 프레임에 추가
adv_df = pd.DataFrame({
    'pca_0': X_test_adv_pca[:, 0],
    'pca_1': X_test_adv_pca[:, 1],
    'label': y_test,
    'prediction': y_test_adv_pred,
    'type': 'adversarial'
})

# 두 데이터 프레임 병합
visualization_df = pd.concat([visualization_df, adv_df])

# 시각화
plt.figure(figsize=(12, 8))

# 원본 데이터 (라벨에 따라 색상 구분)
originals_0 = visualization_df[(visualization_df['type'] == 'original') & (visualization_df['label'] == 0)]
originals_1 = visualization_df[(visualization_df['type'] == 'original') & (visualization_df['label'] == 1)]
plt.scatter(originals_0['pca_0'], originals_0['pca_1'], c='blue', marker='o', label='Original Label 0', alpha=0.6)
plt.scatter(originals_1['pca_0'], originals_1['pca_1'], c='red', marker='o', label='Original Label 1', alpha=0.6)

# 적대적 샘플 (라벨에 따라 색상 구분)
adversarials_0 = visualization_df[(visualization_df['type'] == 'adversarial') & (visualization_df['label'] == 0)]
adversarials_1 = visualization_df[(visualization_df['type'] == 'adversarial') & (visualization_df['label'] == 1)]
plt.scatter(adversarials_0['pca_0'], adversarials_0['pca_1'], c='blue', marker='x', label='Adversarial Label 0', alpha=0.6)
plt.scatter(adversarials_1['pca_0'], adversarials_1['pca_1'], c='red', marker='x', label='Adversarial Label 1', alpha=0.6)

# 레이블 설정 및 범례 추가
plt.xlabel('PCA Component 0')
plt.ylabel('PCA Component 1')
plt.title('Original vs Adversarial Samples in PCA-reduced space')
plt.legend()
plt.show()

# 새로운 평가 메트릭스
new_metrics = {
    "initial_accuracy": initial_accuracy,
    "adv_accuracy": adv_accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "new_adv_accuracy": new_adv_accuracy,
    "new_precision": new_precision,
    "new_recall": new_recall,
    "new_f1": new_f1
}

print(new_metrics)
