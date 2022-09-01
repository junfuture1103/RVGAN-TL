import shap
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import minmax_scale

import sys
import os
import pickle

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import src

# file_name = 'creditcard.csv'
file_name = 'test.csv'

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

plt.rc('font', family='Malgun Gothic')
import matplotlib

lgb = LGBMClassifier(random_state=0, n_estimators=500, learning_rate=0.001, max_depth= 15, min_child_samples= 10)
lgb.fit(X_train, y_train)

matplotlib.rcParams['axes.unicode_minus'] = False

plt.clf()
explainer = shap.TreeExplainer(lgb)
shap_values = explainer.shap_values(X_test)

print("SHAP_VALUES : ")
print(type(shap_values))

with open('get_shap_explainer.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    pickle.dump(explainer, file)
file.close()
with open('get_shap_shap_values.p', 'wb') as file2:  
    pickle.dump(shap_values, file2)
file.close()

# load
with open('get_shap_explainer.p', 'rb') as f:
    explainer = pickle.load(f)
f.close()
# load
with open('get_shap_shap_values.p', 'rb') as f:
    shap_values = pickle.load(f)
f.close()

# shap_values = shap.Explanation(shap_values)

# shap.plots.waterfall(shap_values)
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0])
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values[id].values[:,0],feature_names=X_test.columns)
shap.summary_plot(shap_values)
# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0], show=True, matplotlib=False)
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[[0]], show=False, matplotlib=True)
plt.savefig("point_shape.pdf", format='pdf', dpi=1000, bbox_inches='tight')