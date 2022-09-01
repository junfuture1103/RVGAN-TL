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
    # RandomOverSampler,
    # SMOTE,
    # ADASYN,
    # BorderlineSMOTE,
]

GAN_MODELS = [
    'orgin',
#     src.gans.FDGAN,
#     src.gans.FDGANS,
#     src.gans.FDGANC,
#     src.gans.GAN,
#     src.gans.WGAN,
#     src.gans.WGANGP,
#     src.gans.SNGAN,
# ]
]

if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    src.utils.prepare_dataset(DATASET)
    dataset = src.datasets.FullDataset(training=True)

    raw_x, raw_y = dataset[:]
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    embedded_x = TSNE(
    # learning_rate='auto',
    init='random',
    random_state=src.config.seed,
    ).fit_transform(raw_x)
    result['Original'] = [embedded_x, raw_y]

    sns.set_style('white')
    fig = plt.plot()
    value = result['Original']
       
    majority = []
    minority = []
    
    for i, j in zip(value[0], value[1]):
        if j == 0:
            majority.append(i)
        elif j == 1:
            minority.append(i)
            
    minority = np.array(minority)
    majority = np.array(majority)
    sns.scatterplot(
        x=majority[:, 0],
        y=majority[:, 1],
        alpha=0.5,
        label='majority',
    )
    sns.scatterplot(
        x=minority[:, 0],
        y=minority[:, 1],
        alpha=0.6,
        s=10,
        label='minority',
    )
    # axe.get_legend().remove()

    # fig.set_size_inches(18, 10)
    # fig.set_dpi(100)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.savefig(src.config.path.test_results / 'all_distribution.jpg')
    plt.savefig(src.config.path.test_results / "shap_t-SNE.pdf", format='pdf', dpi=1000, bbox_inches='tight')

    plt.show()
