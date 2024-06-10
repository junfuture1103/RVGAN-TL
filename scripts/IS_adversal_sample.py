import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
file_path = '../data/arff/glass-0-1-2-3_vs_4-5-6.dat_cleaned.arff'
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
print(f'Initial accuracy: {accuracy_score(y_test, y_pred)}')

# 적대적 샘플 생성 (여기서는 간단한 예로, 실제로는 Gradient-based methods 사용)
epsilon = 0.1
X_test_adv = X_test.copy()
X_test_adv += epsilon * np.sign(np.random.randn(*X_test.shape))

# 적대적 샘플에 대한 모델 성능 평가
y_pred_adv = model.predict(X_test_adv)
print(f'Accuracy on adversarial samples: {accuracy_score(y_test, y_pred_adv)}')

# 적대적 샘플 출력
print("Original samples:")
print(X_test.head())
print("\nAdversarial samples:")
print(X_test_adv.head())
