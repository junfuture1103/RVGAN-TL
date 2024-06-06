import pandas as pd
from scipy.io import arff

# 파일 경로 설정
file_path = '../mcc_classifier-master/ecoli2.dat'

# ARFF 파일을 읽고 헤더에서 문제를 일으키는 줄을 주석 처리
with open(file_path, 'r') as file:
    lines = file.readlines()

# 문제를 일으키는 줄 주석 처리
processed_lines = []
for line in lines:
    if "@inputs" in line or "@outputs" in line:
        processed_lines.append(f"% {line}")
    else:
        processed_lines.append(line)

# 처리된 내용을 새 파일에 저장
processed_file_path = '../data/arff/ecoli2_cleaned.arff'
with open(processed_file_path, 'w') as file:
    file.writelines(processed_lines)

# 처리된 ARFF 파일을 불러오기
data, meta = arff.loadarff(processed_file_path)

# pandas DataFrame으로 변환
df = pd.DataFrame(data)

# 각 클래스의 인스턴스 개수 계산
class_counts = df['Class'].value_counts()

IR = class_counts.max() / class_counts.min()

# 결과 출력
class_counts_df = class_counts.to_frame().reset_index()
class_counts_df.columns = ['Class', 'Count']

print(class_counts_df)
print()

print("Class Counts:\n", class_counts)
print("Imbalance Ratio (IR):", IR)
