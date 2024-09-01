from math import sqrt


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
# Add 
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def RandomForest(x_train, y_train, x_test, y_test):
    # modeling
    # Add GBM LGBM ... etc
    model_rf = RandomForestClassifier(n_estimators = 15)
    # train
    model_rf.fit(x_train, y_train)

    # predict
    y_pred = model_rf.predict(x_test) 
    # validation
    y_real = y_test
 
    tn, fp, fn, tp = confusion_matrix(
        y_true=y_real,
        y_pred=y_pred,
    ).ravel()

    accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
    precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
    recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
    specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

    f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
    g_mean = round(sqrt(recall * specificity), 4)

    auc = round(roc_auc_score(
        y_true=y_real,
        y_score=y_pred,
    ),4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)
    print('g_mean : ', g_mean)
    print('auc : ', auc)

    return


def LGBM(x_train, y_train, x_test, y_test):
    # modeling
    model_lgbm = lgb.LGBMClassifier(n_estimators=15, force_col_wise= True)
    # train
    model_lgbm.fit(x_train, y_train)

    # predict
    y_pred = model_lgbm.predict(x_test) 
    # validation
    y_real = y_test
 
    tn, fp, fn, tp = confusion_matrix(
        y_true=y_real,
        y_pred=y_pred,
    ).ravel()

    accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
    precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
    recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
    specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

    f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
    g_mean = round(sqrt(recall * specificity), 4)

    auc = round(roc_auc_score(
        y_true=y_real,
        y_score=y_pred,
    ),4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)
    print('g_mean : ', g_mean)
    print('auc : ', auc)

    return


from sklearn.ensemble import IsolationForest

# def IForest(x_train, y_train, x_test, y_test):

#     # Isolation Forest 모델 생성 및 학습
#     iso_forest = IsolationForest(contamination=0.1, random_state=42)
#     iso_forest.fit(x_train)

#     # 이상값 탐지
#     y_pred = iso_forest.predict(x_test)
#     # -1, 1로 나오는 y_pred 값을 0, 1로 변환
#     y_pred = [0 if i == 1 else 1 for i in y_pred]
#     y_real = y_test
 
#     tn, fp, fn, tp = confusion_matrix(
#         y_true=y_real,
#         y_pred=y_pred,
#     ).ravel()

#     accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
#     precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
#     recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
#     specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

#     f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
#     g_mean = round(sqrt(recall * specificity), 4)

#     auc = round(roc_auc_score(
#         y_true=y_real,
#         y_score=y_pred,
#     ),4)

#     print('Accuracy : ', accuracy)
#     print('Precision : ', precision)
#     print('Recall : ', recall)
#     print('f1-score : ', f1)
#     print('g_mean : ', g_mean)
#     print('auc : ', auc)

#     return

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:48:53 2020

@author: mq20197379

This demo is to show how to use the LSHiForest model for anomaly detection. LSHiForest can use several types of distance metrics.
The IsolationForest model from scikit-learn is used for comparison. Note that this model is a special case of LSHiForest with a standardized L1 distance.

The features in the 'glass.csv' data set has been preprocessed with standardization. The anomaly data instances are with label '-1', while the normal data instances are with label '1'.

"""
import pandas as pd
import numpy as np
import sys

from sklearn.metrics import roc_auc_score

from src.detectors import LSHiForest

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(x, y_pred, title, file_name):
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x)

    plt.figure(figsize=(10, 8))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True)
    
    # Save in different formats
    plt.savefig(f'{file_name}.pdf')
    plt.savefig(f'{file_name}.svg')
    plt.savefig(f'{file_name}.png')
    plt.show()

def IForest(x_train, y_train, x_test, y_test):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(x_train)

    y_pred = iso_forest.predict(x_test)
    y_pred = [0 if i == 1 else 1 for i in y_pred]
    y_real = y_test

    # t-SNE plot
    plot_tsne(x_test, y_pred, 't-SNE of IForest Predictions', 'IForest_tsne')

    tn, fp, fn, tp = confusion_matrix(
        y_true=y_real,
        y_pred=y_pred,
    ).ravel()

    accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
    precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
    recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
    specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

    f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
    g_mean = round(sqrt(recall * specificity), 4)

    auc = round(roc_auc_score(
        y_true=y_real,
        y_score=y_pred,
    ),4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)
    print('g_mean : ', g_mean)
    print('auc : ', auc)

    return

def LSHIForest(x_train, y_train, x_test, y_test):
    iso_forest = LSHiForest()
    iso_forest.fit(x_train)

    y_pred = iso_forest.decision_function(x_test)
    is_inlier = np.ones_like(y_pred, dtype=int)
    is_inlier[y_pred < 0] = -1

    y_pred = is_inlier 
    y_pred = [0 if i == 1 else 1 for i in y_pred]
    y_real = y_test

    # t-SNE plot
    plot_tsne(x_test, y_pred, 't-SNE of LSHIForest Predictions', 'LSHIForest_tsne')

    tn, fp, fn, tp = confusion_matrix(
        y_true=y_real,
        y_pred=y_pred,
    ).ravel()

    accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
    precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
    recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
    specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

    f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
    g_mean = round(sqrt(recall * specificity), 4)

    auc = round(roc_auc_score(
        y_true=y_real,
        y_score=y_pred,
    ),4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)
    print('g_mean : ', g_mean)
    print('auc : ', auc)

    return


def plot_tsne(x, y_pred, title, file_name):
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x)

    plt.figure(figsize=(10, 8))
    
    # 스케일 조정된 값을 사용하여 t-SNE 결과를 시각화
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_pred, cmap='Spectral', s=50, alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True)
    
    # Save in different formats
    plt.savefig(f'{file_name}.pdf')
    plt.savefig(f'{file_name}.svg')
    plt.savefig(f'{file_name}.png')
    plt.show()

def LSHIForestTSNE(x_train, y_train, x_test, y_test):
    iso_forest = LSHiForest()
    iso_forest.fit(x_train)

    # decision_function의 출력을 스케일링
    y_pred = iso_forest.decision_function(x_test)
    
    # MinMaxScaler로 스케일링하여 값의 분포를 0과 1 사이로 조정
    scaler = MinMaxScaler()
    y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

    is_inlier = np.ones_like(y_pred_scaled, dtype=int)
    is_inlier[y_pred_scaled < 0.5] = -1

    y_pred_final = is_inlier 
    y_pred_final = [0 if i == 1 else 1 for i in y_pred_final]
    y_real = y_test

    # t-SNE plot
    plot_tsne(x_test, y_pred_scaled, 't-SNE of LSHIForest Predictions', 'LSHIForest_tsne')

    tn, fp, fn, tp = confusion_matrix(
        y_true=y_real,
        y_pred=y_pred_final,
    ).ravel()

    accuracy = round(sum(y_pred_final == y_real) / len(y_pred_final), 4)
    precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
    recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
    specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

    f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
    g_mean = round(sqrt(recall * specificity), 4)

    auc = round(roc_auc_score(
        y_true=y_real,
        y_score=y_pred_final,
    ),4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)
    print('g_mean : ', g_mean)
    print('auc : ', auc)

    return
