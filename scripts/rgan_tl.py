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
    sys.stdout = open('stdout1.txt', 'w')
    print('Started testing RGAN-TL Classifier')

    # 피클 파일에서 데이터셋 로드
    with open('junganc_dataset_forshap.p', 'rb') as file:
        jungan_dataset = pickle.load(file)

    # 데이터셋을 DataFrame으로 변환
    df = pd.DataFrame(jungan_dataset.samples)
    df['Class'] = jungan_dataset.labels

    # CSV 파일로 저장
    csv_file_name = 'junganc_dataset.csv'
    df.to_csv(csv_file_name, index=False)

    print(f"Dataset saved to {csv_file_name}")

    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)
    
    # print("============ LGBM ============")
    # src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ START SMOTE ============")
    # smote = SMOTE(random_state=42)
    # X_train_resampled, Y_train_resampled = smote.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    # print("============ DONE SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ START SMOTE ============")
    # ada = ADASYN(random_state=42)
    # X_train_resampled, Y_train_resampled = ada.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    # print("============ DONE SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ START SMOTE ============")
    # ros = RandomOverSampler(random_state=42)
    # X_train_resampled, Y_train_resampled = ros.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    # print("============ DONE SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ START SMOTE ============")
    # ros = BorderlineSMOTE(random_state=42)
    # X_train_resampled, Y_train_resampled = ros.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    # print("============ DONE SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

    # print("============ START IForest ============")
    # src.jun_classifier.IForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ DONE IForest ============")


    # print("============ START LSHIForest ============")
    # src.jun_classifier.LSHIForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ DONE LSHIForest ============")

    
    sys.stdout.close()
    sys.stdout = open('stdout2_gan_result.txt', 'w')
    # jungan_dataset = src.utils.get_gan_dataset(src.gans.JUNWGANGP())
    
    # gan_dataset = src.utils.get_gan_dataset(src.gans.GAN())
    # wgan_dataset = src.utils.get_gan_dataset(src.gans.WGAN())
    # wgangp_dataset = src.utils.get_gan_dataset(src.gans.WGANGP())
    # sngan_dataset = src.utils.get_gan_dataset(src.gans.SNGAN())

    jungan_dataset = src.utils.get_jgan_dataset(src.gans.JUNGAN())
    
    with open('junganc_dataset_forshap.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
        # pickle.dump(gan_dataset, file)
        # pickle.dump(wgan_dataset, file)
        # pickle.dump(wgangp_dataset, file)
        # pickle.dump(sngan_dataset, file)
        pickle.dump(jungan_dataset, file)

    ############ GAN ############
    # print("============ RF ============")
    # src.jun_classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ LGBM ============")
    # src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ RF with SMOTE ============")
    # src.jun_classifier.RandomForest(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

    # print("============ LGBM with SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    
    ############ GAN ############
    # print("============ LGBM with GAN ============")
    # src.jun_classifier.LGBM(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ LGBM with WGAN ============")
    # src.jun_classifier.LGBM(wgan_dataset.samples, wgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ LGBM with WGANGP ============")
    # src.jun_classifier.LGBM(wgangp_dataset.samples, wgangp_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ LGBM with SNGANs ============")
    # src.jun_classifier.LGBM(sngan_dataset.samples, sngan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

    # print("============ RF with GAN ============")
    # src.jun_classifier.RandomForest(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ RF with WGAN ============")
    # src.jun_classifier.RandomForest(wgan_dataset.samples, wgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ RF with WGANGP ============")
    # src.jun_classifier.RandomForest(wgangp_dataset.samples, wgangp_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ RF with SNGANs ============")
    # src.jun_classifier.RandomForest(sngan_dataset.samples, sngan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

    # ############ JUNGAN ############
    # print("============ LGBM with JUNGAN ============")
    # src.jun_classifier.LGBM(jungan_dataset.samples, jungan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    sys.stdout.close()

    # sys.stdout = open('stdout3.txt', 'w')
    # print("============ LGBM with RVJUNGANC ============")
    # junganc_dataset = src.utils.get_jgan_dataset(src.gans.JUNGANC())
 
    # with open('junganc_dataset_rv.p', 'wb') as file2:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    #     pickle.dump(junganc_dataset, file2)

    # src.jun_classifier.LGBM(junganc_dataset.samples, junganc_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # sys.stdout.close()
    # print("============ LGBM with RVSNGANs ============")
    # src.jun_classifier.LGBM(rvsngan_dataset.samples, rvsngan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

    # lgbm_classifier = src.lgbm.LGBM()
    # lgbm_classifier.fit(rgan_dataset)
    # lgbm_classifier.test(test_dataset)
    # | tee output.txt
    # print("============ LGBM with RGAN ============")
    # for name, value in lgbm_classifier.metrics.items():
    #     print(f'{name:<15}:{value:>10.4f}')

    # print('Started testing Original Classifier')
    # original_classifier = src.classifier.Classifier('Original')
    # original_classifier.fit(full_dataset)
    # for name, value in original_classifier.metrics.items():
    #     print(f'{name:<15}:{value:>10.4f}')