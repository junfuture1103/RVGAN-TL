import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE

FILE_NAME = 'creditcard.csv'

if __name__ == '__main__':
    print('Started testing RGAN-TL Classifier')
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)

    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    
    rgan_dataset = src.utils.get_jgan_dataset(src.gans.GAN())

    # src.classifier.Classifier(gan_dataset)
    print("============ RF ============")
    src.jun_classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ LGBM ============")
    src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ RF with RGAN ============")
    src.jun_classifier.RandomForest(rgan_dataset.samples, rgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ LGBM with RGAN ============")
    src.jun_classifier.LGBM(rgan_dataset.samples, rgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

    print("============ RF with SMOTE ============")
    src.jun_classifier.RandomForest(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

    print("============ LGBM with SMOTE ============")
    src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

   
    # print("============ LGBM with SMOTE ============")
    # src.regression.LGBM(X_train_resampled, y_train_resampled, test_samples, test_labels)


    # tl_classifier = src.tr_ada_boost.TrAdaBoost()
    # tl_classifier.fit(rgan_dataset, full_dataset)
    # tl_classifier.test(test_dataset)

    # for name, value in tl_classifier.metrics.items():
    #     print(f'{name:<15}:{value:>10.4f}')

    # print('Started testing Original Classifier')
    # original_classifier = src.classifier.Classifier('Original')
    # original_classifier.fit(full_dataset)
    # for name, value in original_classifier.metrics.items():
    #     print(f'{name:<15}:{value:>10.4f}')
