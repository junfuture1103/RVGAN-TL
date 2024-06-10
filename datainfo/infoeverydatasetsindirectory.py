import os
import pandas as pd
from scipy.io import arff

# 데이터셋이 저장된 디렉토리 경로 설정
directory_path = '../mcc_classifier-master/'
output_directory = '../results/'

# 결과를 저장할 디렉토리 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 결과를 저장할 파일 경로 설정
result_file_path = os.path.join(output_directory, 'results.txt')

# 결과 파일 초기화
with open(result_file_path, 'w') as result_file:
    result_file.write("")

# 디렉토리 내의 모든 파일에 대해 반복
for filename in os.listdir(directory_path):
    if filename.endswith(".dat"):  # 확장자가 .dat인 파일만 처리
        file_path = os.path.join(directory_path, filename)
        
        # ARFF 파일을 읽고 헤더에서 문제를 일으키는 줄을 제거
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 문제를 일으키는 줄 제거
        processed_lines = []
        for line in lines:
            if "@inputs" in line or "@outputs" in line or "@input" in line or "@output" in line:
                continue  # 해당 줄을 무시
            else:
                processed_lines.append(line)

        # 처리된 내용을 새 파일에 저장
        processed_file_path = f'../data/arff/{filename}_cleaned.arff'
        with open(processed_file_path, 'w') as file:
            file.writelines(processed_lines)

        # 처리된 ARFF 파일을 불러오기
        try:
            data, meta = arff.loadarff(processed_file_path)

            # pandas DataFrame으로 변환
            df = pd.DataFrame(data)

            # 클래스 라벨을 문자열로 변환
            df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))

            # 클래스 라벨을 숫자로 변환
            class_mapping = {label: idx for idx, label in enumerate(df['Class'].unique())}
            df['Class'] = df['Class'].map(class_mapping)

            # 각 클래스의 인스턴스 개수 계산
            class_counts = df['Class'].value_counts()

            IR = class_counts.max() / class_counts.min()

            # 결과 출력
            class_counts_df = class_counts.to_frame().reset_index()
            class_counts_df.columns = ['Class', 'Count']

            result_text = f"File: {filename}\n"
            result_text += class_counts_df.to_string(index=False) + "\n\n"
            result_text += "Class Counts:\n" + class_counts.to_string() + "\n"
            result_text += "Imbalance Ratio (IR): " + str(IR) + "\n"
            result_text += "="*50 + "\n"  # 구분선을 출력하여 각 파일의 결과를 구분

            # 결과를 파일에 저장
            with open(result_file_path, 'a') as result_file:
                result_file.write(result_text)

        except Exception as e:
            error_text = f"Error processing file {filename}: {e}\n"
            with open(result_file_path, 'a') as result_file:
                result_file.write(error_text)

print(f"Results have been saved to {result_file_path}")
