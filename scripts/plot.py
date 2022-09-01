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

plt.clf()
file_name = 'creditcard.csv'
# file_name = 'test.csv'

file_path = src.config.path.datasets / file_name
input = pd.read_csv(file_path)

samples = input.loc[:, 'V1' : 'V28']
labels = input.loc[:, 'Class']

X = samples
y = labels

# train XGBoost model
# X,y = shap.datasets.adult()
# model = xgboost.XGBClassifier().fit(X, y)
model = LGBMClassifier().fit(X, y)
# model = LGBMClassifier(random_state=0, n_estimators=500, learning_rate=0.001, max_depth= 15, min_child_samples= 10).fit(X,y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

plt.clf()
# shap.summary_plot(shap_values)
# plt.savefig("point_sh1.pdf", format='pdf', dpi=1000, bbox_inches='tight')
# plt.clf()
# shap.summary_plot(shap_values, plot_type = "bar", show=False)
# plt.savefig("point_sh2.pdf", format='pdf', dpi=1000, bbox_inches='tight')
# plt.clf()
# shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[0], max_display=20)
plt.savefig("point_sh3.pdf", format='pdf', dpi=1000, bbox_inches='tight')