import shap
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import minmax_scale

import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src 

file_name = 'creditcard.csv'

file_path = src.config.path.datasets / file_name
input = pd.read_csv(file_path)


samples = input.loc[:, 'V1' : 'Amount']
labels = input.loc[:, 'Class']

# normalize samples -> min : 0 max : 1
samples = minmax_scale(samples.astype('float32'))
labels = labels.astype('int')

y_feature = labels
X_feature = samples

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_feature, y_feature, test_size=0.2, 
random_state =50)

from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from matplotlib import font_manager

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import matplotlib

lgb = LGBMClassifier(random_state=0, n_estimators=500, learning_rate=0.001, max_depth= 15, min_child_samples= 10)
lgb.fit(X_train, y_train)

matplotlib.rcParams['axes.unicode_minus'] = False

plt.clf()
explainer = shap.TreeExplainer(lgb)
shap_values = explainer.shap_values(X_test)

# shap.waterfall(shap_values[0])
shap.summary_plot(shap_values, X_test, show=False)
# shap.summary_plot(shap_values, X_test, plot_type = "bar", show=False)
# X_test = pd.DataFrame(X_test)
# shap.initjs()
# # shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[[0]], show=False, matplotlib=True)
# shap.force_plot(explainer.expected_value[0],X_test.iloc[0,:])
plt.savefig("point_shap.pdf", format='pdf', dpi=1000, bbox_inches='tight')