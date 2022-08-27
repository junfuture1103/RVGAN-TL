from math import sqrt, log, ceil

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

import src.config.tr_ada_boost
from src import config
from src.config.tr_ada_boost import classifiers as n_classifier
from src.datasets import BasicDataset
from lightgbm import LGBMClassifier

class LGBM:
    def __init__(self):
        self.metrics = {
            'Accuracy': .0,
            'Precision': .0,
            'Recall': .0,
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
        }
        self.model = LGBMClassifier(random_state=0, n_estimators=500, learning_rate=0.05, max_depth= 15, min_child_samples= 10)

    def fit(self, src_dataset: BasicDataset):
        src_dataset.to('cpu')
        src_x, src_y = src_dataset.samples.numpy(), src_dataset.labels.numpy()
        self.model.fit(src_x, src_y)

    def predict(self, x: torch.Tensor):
        x = x.cpu().numpy()
        prediction = self.model.predict(x)
        return torch.from_numpy(prediction).to(src.config.device)

    def test(self, test_dataset: BasicDataset):
        with torch.no_grad():
            x, label = test_dataset.samples.cpu(), test_dataset.labels.cpu()
            predicted_label = self.predict(x).cpu()
            tn, fp, fn, tp = confusion_matrix(
                y_true=label,
                y_pred=predicted_label,
            ).ravel()

            accuracy = sum(predicted_label == label) / len(predicted_label)
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            specificity = tn / (tn + fp) if tn + fp != 0 else 0

            f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            g_mean = sqrt(recall * specificity)

            auc = roc_auc_score(
                y_true=label,
                y_score=predicted_label,
            )
            self.metrics['Accuracy'] = accuracy
            self.metrics['Precision'] = precision
            self.metrics['Recall'] = recall
            self.metrics['F1'] = f1
            self.metrics['G-Mean'] = g_mean
            self.metrics['AUC'] = auc
