import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE

FILE_NAME = 'creditcard.csv'

if __name__ == '__main__':
    sys.stdout = open('stdout1.txt', 'w')
    print('Started testing RGAN-TL Classifier')
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)

    print("============ START SMOTE ============")
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    print("============ DONE SMOTE ============")
    
    sys.stdout.close()
    sys.stdout = open('stdout2.txt', 'w')
    gan_dataset = src.utils.get_gan_dataset(src.gans.GAN())
    wgan_dataset = src.utils.get_gan_dataset(src.gans.WGAN())
    wgangp_dataset = src.utils.get_gan_dataset(src.gans.WGANGP())
    sngan_dataset = src.utils.get_gan_dataset(src.gans.SNGAN())

    # jungan_dataset = src.utils.get_jgan_dataset(src.gans.JUNGAN())
    jungans_dataset = src.utils.get_gan_dataset(src.gans.JUNGANS())
    # junganc_dataset = src.utils.get_jgan_dataset(src.gans.JUNGANC())
    junganc_dataset = src.utils.get_gan_dataset(src.gans.JUNGANC())
    # rvsngan_dataset = src.utils.get_jgan_dataset(src.gans.RVSNGAN())
    
    ############ GAN ############
    print("============ RF ============")
    src.jun_classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ LGBM ============")
    src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ RF with SMOTE ============")
    src.jun_classifier.RandomForest(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

    print("============ LGBM with SMOTE ============")
    src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    
    ############ GAN ############
    print("============ LGBM with GAN ============")
    src.jun_classifier.LGBM(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ LGBM with WGAN ============")
    src.jun_classifier.LGBM(wgan_dataset.samples, wgan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    print("============ LGBM with WGANGP ============")
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

    sys.stdout.close()
    # ############ JUNGAN ############
    # print("============ LGBM with JUNGAN ============")
    # src.jun_classifier.LGBM(jungan_dataset.samples, jungan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)

    # print("============ LGBM with JUNGANS ============")
    src.jun_classifier.LGBM(jungans_dataset.samples, jungans_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ LGBM with JUNGANC ============")
    src.jun_classifier.LGBM(junganc_dataset.samples, junganc_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    sys.stdout.close()

    sys.stdout = open('stdout3.txt', 'w')
    print("============ LGBM with RVJUNGANC ============")
    junganc_dataset = src.utils.get_jgan_dataset(src.gans.JUNGANC())
    src.jun_classifier.LGBM(junganc_dataset.samples, junganc_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
    
    sys.stdout.close()
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