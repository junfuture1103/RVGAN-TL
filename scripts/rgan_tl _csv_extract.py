import sys
import os
import pickle
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

FILE_NAME = 'creditcard.csv'

if __name__ == '__main__':
    sys.stdout = open('creditcard.txt', 'w')
    # print('Started testing RGAN-TL Classifier')

    # # 피클 파일에서 데이터셋 로드
    # with open('junganc_dataset_forshap.p', 'rb') as file:
    #     jungan_dataset = pickle.load(file)
    #     full_dataset = pickle.load(file)

    # # 데이터셋을 DataFrame으로 변환
    # df = pd.DataFrame(jungan_dataset.samples)
    # df['Class'] = jungan_dataset.labels

    # # CSV 파일로 저장
    # csv_file_name = 'junganc_dataset.csv'
    # df.to_csv(csv_file_name, index=False)

    # print(f"Dataset saved to {csv_file_name}")


    # # 데이터셋을 DataFrame으로 변환
    # df = pd.DataFrame(full_dataset.samples)
    # df['Class'] = full_dataset.labels

    # # CSV 파일로 저장
    # csv_file_name = 'original_dataset.csv'
    # df.to_csv(csv_file_name, index=False)

    # print(f"Dataset saved to {csv_file_name}")
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)

    # # 피클 파일에서 데이터셋 로드
    # with open('junganc_dataset_forshap.p', 'rb') as file:
    #     jungan_dataset = pickle.load(file)

    # # 데이터셋을 DataFrame으로 변환
    # df = pd.DataFrame(jungan_dataset.samples)
    # df['Class'] = jungan_dataset.labels

    # # CSV 파일로 저장
    # csv_file_name = 'junganc_dataset.csv'
    # df.to_csv(csv_file_name, index=False)

    # print(f"Dataset saved to {csv_file_name}")
    
    # jungan_dataset = src.utils.get_jgan_dataset(src.gans.JUNGAN())
    
    # with open('junganc_dataset_forshap.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    #     pickle.dump(jungan_dataset, file)
    #     pickle.dump(full_dataset, file)