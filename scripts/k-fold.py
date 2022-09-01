import sys
import os
import pickle
import torch

from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE

FILE_NAME = 'creditcard.csv'
from sklearn.model_selection import KFold

    
if __name__ == '__main__':
    sys.stdout = open('stdout_kfold2.txt', 'w')

    print('Started testing RGAN-TL Classifier')
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)

    gan_datasets = []

    with open(file='junganc_dataset.p', mode='rb') as f:
        gan_dataset=pickle.load(f)
        gan_datasets.append(gan_dataset)
        wgan_dataset=pickle.load(f)
        gan_datasets.append(wgan_dataset)
        wgangp_dataset=pickle.load(f)
        gan_datasets.append(wgangp_dataset)
        sngan_dataset=pickle.load(f)
        gan_datasets.append(sngan_dataset)
        jungans_dataset=pickle.load(f)
        gan_datasets.append(jungans_dataset)
        junganc_dataset=pickle.load(f)
        gan_datasets.append(junganc_dataset)
        jungan_dataset=pickle.load(f)
        gan_datasets.append(jungan_dataset)

    # split 개수, 셔플 여부 및 seed 설정
    kf = KFold(n_splits = 5, shuffle = True, random_state = 50)

    result = dict()
    tmp_result = []
    for gan in gan_datasets:
        print("============ ============")
        X = gan.samples
        y = gan.labels

        # split 개수 스텝 만큼 train, test 데이터셋을 매번 분할
        tmp_result = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            tmp = src.jun_classifier.LGBM(X_train, y_train, src.datasets.test_samples, src.datasets.test_labels)           
            print(tmp)
            tmp_result.append(tmp)

        gan_result = [
            .0,.0,.0,.0,.0,.0
        ]
        
        for re in tmp_result:
            for i in range(0,len(gan_result)):
                gan_result[i] += re[i]

        for i in range(0,len(gan_result)):
            gan_result[i] = round(gan_result[i] / len(tmp_result) , 4)

        print(gan_result)    
            
    sys.stdout.close()