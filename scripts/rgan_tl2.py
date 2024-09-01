import sys
import os
import pickle
import glob

from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

# FILE_NAME = 'creditcard.csv'
ARFF_FILE_NAME = '../data/arff/vehicle0.dat_cleaned.arff'

directory = '../data/testarff/'
arff_files = glob.glob(os.path.join(directory, '*.arff'))

if __name__ == '__main__':
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('start ', current_time)
    # 2. 디렉터리 생성
    directory_name = f'test_{current_time}'
    os.makedirs(directory_name, exist_ok=True)
    
    for arff_file in arff_files:
        # print('now processing : ', arff_file)
        
        file_name = os.path.basename(arff_file)

        # 확장자 제거
        file_name_without_ext = os.path.splitext(file_name)[0]
        # sys.stdout = open('stdout_traditional_{}.txt'.format(file_name_without_ext), 'w')
        sys.stdout = open(f'{directory_name}/stdout_traditional_{file_name_without_ext}.txt', 'w')

        print('Started testing RGAN-TL Classifier : '+ arff_file)
        src.utils.set_random_state()

        # src.utils.prepare_dataset(FILE_NAME)
        src.utils.prepare_arff_dataset(arff_file)
        
        full_dataset = src.datasets.FullDataset()
        test_dataset = src.datasets.FullDataset(training=False)
        
        print("============ LGBM ============")
        src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ START SMOTE ============")
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
        print("============ DONE SMOTE ============")
        src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ START ADASYN ============")
        ada = ADASYN(random_state=42)
        X_train_resampled, Y_train_resampled = ada.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
        print("============ DONE ADASYN ============")
        src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ START ROS ============")
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, Y_train_resampled = ros.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
        print("============ DONE ROS ============")
        src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ START BorderlineSMOTE ============")
        ros = BorderlineSMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = ros.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
        print("============ DONE BorderlineSMOTE ============")
        src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ START IForest ============")
        src.jun_classifier.IForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ DONE IForest ============")

        sys.stdout.close()
        # sys.stdout = open('stdout_gan_{}.txt'.format(file_name_without_ext), 'w')
        sys.stdout = open(f'{directory_name}/stdout_gan_{file_name_without_ext}.txt', 'w')

        sngan_dataset = src.utils.get_gan_dataset(src.gans.SNGAN())
        gan_dataset = src.utils.get_gan_dataset(src.gans.GAN())
        wgan_dataset = src.utils.get_gan_dataset(src.gans.WGAN())
        wgangp_dataset = src.utils.get_gan_dataset(src.gans.WGANGP())
        jungan_dataset = src.utils.get_jgan_dataset(src.gans.JUNGAN())
        
        with open('jungan_dataset_{}.p'.format(file_name_without_ext), 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
            pickle.dump(gan_dataset, file)
            pickle.dump(wgan_dataset, file)
            pickle.dump(wgangp_dataset, file)
            pickle.dump(sngan_dataset, file)
            pickle.dump(jungan_dataset, file)

        ############ GAN ############
        print("============ RF ============")
        src.jun_classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ LGBM ============")
        src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ RF with SMOTE ============")
        src.jun_classifier.RandomForest(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

        print("============ LGBM with SMOTE ============")
        src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
        
        # ############ GAN ############
        print("============ LGBM with GAN ============")
        src.jun_classifier.LGBM(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        # print("============ LGBM with WGAN ============")
        src.jun_classifier.LGBM(wgan_dataset.samples, wgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        # print("============ LGBM with WGANGP ============")
        src.jun_classifier.LGBM(wgangp_dataset.samples, wgangp_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ LGBM with SNGANs ============")
        src.jun_classifier.LGBM(sngan_dataset.samples, sngan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

        print("============ RF with GAN ============")
        src.jun_classifier.RandomForest(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ RF with WGAN ============")
        src.jun_classifier.RandomForest(wgan_dataset.samples, wgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ RF with WGANGP ============")
        src.jun_classifier.RandomForest(wgangp_dataset.samples, wgangp_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ RF with SNGANs ============")
        src.jun_classifier.RandomForest(sngan_dataset.samples, sngan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

        # ############ JUNGAN ############
        print("============ RF with JUNGAN ============")
        src.jun_classifier.RandomForest(jungan_dataset.samples, jungan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        print("============ LGBM with JUNGAN ============")
        src.jun_classifier.LGBM(jungan_dataset.samples, jungan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
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