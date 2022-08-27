import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src

FILE_NAME = 'creditcard.csv'

if __name__ == '__main__':
    print('Started testing RGAN-TL Classifier')
    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)

    print(full_dataset)
    print(test_dataset)

    rgan_dataset = src.utils.get_rgan_dataset(src.gans.RVGAN())
    tl_classifier = src.tr_ada_boost.TrAdaBoost()
    tl_classifier.fit(rgan_dataset, full_dataset)
    tl_classifier.test(test_dataset)
    for name, value in tl_classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')

    print('Started testing Original Classifier')
    original_classifier = src.classifier.Classifier('Original')
    original_classifier.fit(full_dataset)
    for name, value in original_classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')
