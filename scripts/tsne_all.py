import context

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler

import src

DATASET = 'creditcard.csv'
# DATASET = 'test.csv'

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

GAN_MODELS = [
    'orgin',
    # src.gans.FDGAN,
    # src.gans.FDGANS,
    # src.gans.FDGANC,
    # src.gans.GAN,
    # src.gans.WGAN,
    # src.gans.WGANGP,
    # src.gans.SNGAN,
]
import pickle


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

if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    src.utils.prepare_dataset(DATASET)
    dataset = src.datasets.FullDataset(training=True)

    gan_datasets = []

    # with open(file='junganc_dataset.p', mode='rb') as f:
    #     gan_dataset=pickle.load(f)
    #     wgan_dataset=pickle.load(f)
    #     wgangp_dataset=pickle.load(f)
    #     sngan_dataset=pickle.load(f)
    #     jungans_dataset=pickle.load(f)
    #     junganc_dataset=pickle.load(f)
    #     jungan_dataset=pickle.load(f)

    # gan_datasets = [jungan_dataset, jungans_dataset, junganc_dataset, gan_dataset,wgan_dataset,wgangp_dataset,sngan_dataset]
    
    raw_x, raw_y = dataset[:]
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    plot_tsne(raw_x, raw_y,'t-SNE of original', 'ORIGINAL_tsne')

    # idx = 0
    # for M in GAN_MODELS:
    #     src.utils.set_random_state()
    #     if(M=='orgin'):
    #         embedded_x = TSNE(
    #         # learning_rate='auto',
    #         init='random',
    #         random_state=src.config.seed,
    #         ).fit_transform(raw_x)
    #         result['Original'] = [embedded_x, raw_y]
            

    # for M in TRADITIONAL_METHODS:
    #     print("== START {} ==".format(M.__name__))
    #     x, _ = M(random_state=src.config.seed).fit_resample(raw_x, raw_y)
    #     y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    #     embedded_x = TSNE(
    #         # learning_rate='auto',
    #         init='random',
    #         random_state=src.config.seed,
    #     ).fit_transform(x)
    #     result[M.__name__] = [embedded_x, y]

    # sns.set_style('white')
    # fig, axes = plt.subplots(3, 4)
    # for (key, value), axe in zip(result.items(), axes.flat):
    #     if(key != 'Original'):
    #         axe.set(title=key)
    #         majority = []
    #         minority = []
    #         generated_data = []
    #         for i, j in zip(value[0], value[1]):
    #             if j == 0:
    #                 majority.append(i)
    #             elif j == 1:
    #                 minority.append(i)
    #             else:
    #                 generated_data.append(i)
    #         minority = np.array(minority)
    #         majority = np.array(majority)
    #         generated_data = np.array(generated_data)
    #         print(generated_data[:,0])              
    #         sns.scatterplot(
    #             x=majority[:, 0],
    #             y=majority[:, 1],
    #             ax=axe,
    #             alpha=0.5,
    #             label='majority',
    #         )
    #         sns.scatterplot(
    #             x=generated_data[:, 0],
    #             y=generated_data[:, 1],
    #             ax=axe,
    #             alpha=1.0,
    #             label='generated_data',
    #         )
    #         sns.scatterplot(
    #             x=minority[:, 0],
    #             y=minority[:, 1],
    #             ax=axe,
    #             alpha=0.6,
    #             s=10,
    #             label='minority',
    #         )
    #         axe.get_legend().remove()
    #     else:
    #         axe.set(title=key)
    #         majority = []
    #         minority = []
    #         generated_data = []
    #         for i, j in zip(value[0], value[1]):
    #             if j == 0:
    #                 majority.append(i)
    #             elif j == 1:
    #                 minority.append(i)
    #         minority = np.array(minority)
    #         majority = np.array(majority)       
    #         sns.scatterplot(
    #             x=majority[:, 0],
    #             y=majority[:, 1],
    #             ax=axe,
    #             alpha=0.5,
    #             label='majority',
    #         )
    #         sns.scatterplot(
    #             x=minority[:, 0],
    #             y=minority[:, 1],
    #             ax=axe,
    #             alpha=0.6,
    #             s=10,
    #             label='minority',
    #         )
    #         axe.get_legend().remove()

    # fig.set_size_inches(18, 10)
    # fig.set_dpi(100)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # plt.savefig(src.config.path.test_results / 'all_distribution.jpg')
    # plt.savefig(src.config.path.test_results / "shap_t-SNE.pdf", format='pdf', dpi=1000, bbox_inches='tight')

    # plt.show()
