import xgboost
import shap
import pandas as pd
import sys
import os
import pickle

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import src

from lightgbm import LGBMClassifier
file_name = 'creditcard.csv'

file_path = src.config.path.datasets / file_name
input = pd.read_csv(file_path)

samples = input.loc[:, 'V1' : 'Amount']
labels = input.loc[:, 'Class']

# # normalize samples -> min : 0 max : 1
# samples = minmax_scale(samples.astype('float32'))
# labels = labels.astype('int')

y_feature = labels
X_feature = samples

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_feature, y_feature, test_size=0.2, 
random_state =50)

model = LGBMClassifier(random_state=0, n_estimators=500, learning_rate=0.001, max_depth= 15, min_child_samples= 10)
model.fit(X_train, y_train)


with open('lgb_model.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    pickle.dump(model, file)
file.close()

