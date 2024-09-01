from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# 예제 데이터 생성 (2차원 데이터)
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

# Isolation Forest 모델 생성 및 학습
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X)

# 이상값 탐지
y_pred = iso_forest.predict(X)

# 시각화
plt.title("Isolation Forest Anomaly Detection")
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Inliers')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='red', label='Outliers')
plt.legend()
plt.show()
