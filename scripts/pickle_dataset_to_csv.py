import pickle
import pandas as pd
import sys
import os
import pickle
import src

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 피클 파일에서 데이터셋 로드
with open('junganc_dataset_forshap.p', 'rb') as file:
    jungan_dataset = pickle.load(file)
    full_dataset = pickle.load(file)

# 데이터셋을 DataFrame으로 변환
df = pd.DataFrame(jungan_dataset.samples)
df['Class'] = jungan_dataset.labels

# CSV 파일로 저장
csv_file_name = 'junganc_dataset.csv'
df.to_csv(csv_file_name, index=False)

print(f"Dataset saved to {csv_file_name}")


# 데이터셋을 DataFrame으로 변환
df = pd.DataFrame(full_dataset.samples)
df['Class'] = full_dataset.labels

# CSV 파일로 저장
csv_file_name = 'original_dataset.csv'
df.to_csv(csv_file_name, index=False)

print(f"Dataset saved to {csv_file_name}")
