import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 파일 경로 설정
file_path = '../mcc_classifier-master/ecoli2.dat'

# ARFF 파일을 읽고 헤더에서 문제를 일으키는 줄을 주석 처리
with open(file_path, 'r') as file:
    lines = file.readlines()

# 문제를 일으키는 줄 주석 처리
processed_lines = []
for line in lines:
    if "@inputs" in line or "@outputs" in line:
        processed_lines.append(f"% {line}")
    else:
        processed_lines.append(line)

# 처리된 내용을 새 파일에 저장
processed_file_path = '../data/arff/ecoli2_cleaned.arff'
with open(processed_file_path, 'w') as file:
    file.writelines(processed_lines)

# 처리된 ARFF 파일을 불러오기
data, meta = arff.loadarff(processed_file_path)

# pandas DataFrame으로 변환
df = pd.DataFrame(data)

# 클래스 라벨을 문자열로 변환
df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))

# 클래스 라벨을 숫자로 변환
class_mapping = {'negative': 0, 'positive': 1}
df['Class'] = df['Class'].map(class_mapping)

# 각 클래스의 인스턴스 개수 계산
class_counts = df['Class'].value_counts()

IR = class_counts.max() / class_counts.min()

# 결과 출력
class_counts_df = class_counts.to_frame().reset_index()
class_counts_df.columns = ['Class', 'Count']

print(class_counts_df)
print()

print("Class Counts:\n", class_counts)
print("Imbalance Ratio (IR):", IR)

# 특징과 클래스 분리
X = df.drop('Class', axis=1).astype(float)
y = df['Class']

# PCA를 사용하여 데이터의 차원을 축소 (2D로)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KNN 분류기 설정 및 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_pca, y)

# 데이터 시각화
plt.figure(figsize=(10, 7))

# 클래스별 데이터 포인트 색상 설정
colors = {0: 'blue', 1: 'red'}

# 원본 데이터 포인트 시각화
for class_value in class_counts.index:
    subset = X_pca[y == class_value]
    plt.scatter(subset[:, 0], subset[:, 1], c=colors[class_value], label='positive' if class_value == 1 else 'negative', alpha=0.5, edgecolors='w')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KNN Classification of Positive and Negative Classes')
plt.legend()
plt.show()
