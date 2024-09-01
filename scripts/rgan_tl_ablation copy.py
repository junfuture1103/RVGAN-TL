import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

import itertools

FILE_NAME = 'creditcard.csv'
ARFF_FILE_NAME = '../data/arff/ecoli3.dat_cleaned.arff'
if __name__ == '__main__':
    print('Started testing RGAN-TL Classifier')

    src.utils.set_random_state()
    src.utils.prepare_arff_dataset(ARFF_FILE_NAME)
    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)


    # 모든 경우의 수를 위한 설정 값들
    spectral_norm_options = [True, False]
    celu_options = [True, False]
    layernorm_options = [True, False]

    # 모든 경우의 수 생성
    all_ablation_configs = list(itertools.product(spectral_norm_options, celu_options, layernorm_options))

    # 각 ablation 설정에 따른 코드 실행
    for idx, (use_spectral_norm, use_celu, use_layernorm) in enumerate(all_ablation_configs):
        ablation_config = {
            'use_spectral_norm': use_spectral_norm,
            'use_celu': use_celu,
            'use_layernorm': use_layernorm,
            'step_1_layers': [512, 128, 32, 8]
        }
        
        print(f"Running configuration {idx + 1}/{len(all_ablation_configs)}: "
            f"use_spectral_norm={use_spectral_norm}, use_celu={use_celu}, use_layernorm={use_layernorm}")
        
        # just gan True True True
        jungan_dataset = src.utils.get_gan_dataset(src.gans.JUNGAN(ablation_config=ablation_config))
        # jgan True True True
        jungan_dataset = src.utils.get_jgan_dataset(src.gans.JUNGAN(ablation_config=ablation_config))
    
        # 여기에 추가적인 실험 코드 또는 모델 학습 코드를 넣으세요
        # 예를 들어, 모델 학습을 수행할 수 있습니다.
        # model = src.gans.JUNGAN(ablation_config=ablation_config)
        # model._fit()

        # 고유한 파일 이름을 생성하여 pickle로 데이터셋 저장
        filename = f"jungan_dataset_spectral_jgan_{use_spectral_norm}_celu_{use_celu}_layernorm_{use_layernorm}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(jungan_dataset, f)
        
        print(f"Dataset saved to {filename}")

        # 결과를 저장할 파일 이름 설정
        results_filename = f"results_spectral_just_gan_{use_spectral_norm}_celu_{use_celu}_layernorm_{use_layernorm}.txt"
        with open(results_filename, 'w') as f:
            # stdout을 파일로 리다이렉트
            sys.stdout = f
            
            print(f"Running RandomForest and LGBM for configuration {idx + 1}/{len(all_ablation_configs)}:")
            print("=============================================")
            
            # Random Forest 실행
            print("============ RF with GAN ============")
            src.jun_classifier.RandomForest(jungan_dataset.samples, jungan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
            
            # LightGBM 실행
            print("============ LGBM with JUNGAN ============")
            src.jun_classifier.LGBM(jungan_dataset.samples, jungan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)
        
        # stdout을 다시 원래대로 복구
        sys.stdout = sys.__stdout__
        
        print(f"Results saved to {results_filename}")