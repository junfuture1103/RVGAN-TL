import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

# FILE_NAME = 'creditcard.csv'
ARFF_FILE_NAME = '../data/arff/vehicle0.dat_cleaned.arff'

if __name__ == '__main__':
    # sys.stdout = open('stdout1.txt', 'w')

    print('Started testing RGAN-TL Classifier')
    src.utils.set_random_state()
    # src.utils.prepare_dataset(FILE_NAME)
    src.utils.prepare_arff_dataset(ARFF_FILE_NAME)
    
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)
    
    # print("============ LGBM ============")
    src.jun_classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ START SMOTE ============")
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    print("============ DONE SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    