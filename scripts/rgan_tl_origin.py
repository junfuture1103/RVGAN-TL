import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

FILE_NAME = 'creditcard.csv'

# if __name__ == '__main__':
#     sys.stdout = open('creditcard.txt', 'w')
#     print('Started testing RGAN-TL Classifier')

#     src.utils.set_random_state()
#     src.utils.prepare_dataset(FILE_NAME)
#     full_dataset = src.datasets.FullDataset()
#     test_dataset = src.datasets.FullDataset(training=False)

#     print("============ START IForest ============")
#     # src.jun_classifier.IForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
#     print("============ DONE IForest ============")


#     print("============ START LSHIForest ============")
#     src.jun_classifier.LSHIForestTSNE(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
#     print("============ DONE LSHIForest ============")

    
#     sys.stdout.close()
    # sys.stdout = open('stdout2.txt', 'w')
    # # jungan_dataset = src.utils.get_gan_dataset(src.gans.JUNWGANGP())
    
    # gan_dataset = src.utils.get_gan_dataset(src.gans.GAN())
    # wgan_dataset = src.utils.get_gan_dataset(src.gans.WGAN())
    # wgangp_dataset = src.utils.get_gan_dataset(src.gans.WGANGP())
    # sngan_dataset = src.utils.get_gan_dataset(src.gans.SNGAN())

    # jungan_dataset = src.utils.get_jgan_dataset(src.gans.JUNGAN())
    
    # with open('junganc_dataset.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    #     pickle.dump(gan_dataset, file)
    #     pickle.dump(wgan_dataset, file)
    #     pickle.dump(wgangp_dataset, file)
    #     pickle.dump(sngan_dataset, file)

    # ############ GAN ############
    # print("============ RF ============")
    # src.jun_classifier.RandomForest(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ LGBM ============")
    # src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # print("============ RF with SMOTE ============")
    # src.jun_classifier.RandomForest(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)

    # print("============ LGBM with SMOTE ============")
    # src.jun_classifier.LGBM(X_train_resampled, Y_train_resampled, src.datasets.test_samples, src.datasets.test_labels)
    
    # ############ GAN ############
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
    # sys.stdout.close()

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_original(x, y, title, file_name):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_x = tsne.fit_transform(x)

    plt.figure(figsize=(10, 8))

    majority = []
    minority = []

    for i, label in zip(embedded_x, y):
        if label == 0:
            majority.append(i)
        elif label == 1:
            minority.append(i)

    majority = np.array(majority)
    minority = np.array(minority)

    plt.scatter(
        x=majority[:, 0],
        y=majority[:, 1],
        alpha=0.5,
        label='majority',
        c='blue',  # Blue for majority class
    )
    
    plt.scatter(
        x=minority[:, 0],
        y=minority[:, 1],
        alpha=0.6,
        s=10,
        label='minority',
        c='red',  # Red for minority class
    )

    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.savefig(f'{file_name}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{file_name}.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(f'{file_name}.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == '__main__':
    # 데이터셋 준비
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    dataset = src.datasets.FullDataset(training=True)

    raw_x, raw_y = dataset[:]
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    # Original 데이터에 대해 t-SNE 수행 및 시각화
    plot_tsne_original(raw_x, raw_y, 't-SNE of Original Data', 'Original_tsne')
