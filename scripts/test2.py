import pandas as pd

# Breast Cancer Wisconsin (Diagnostic) Dataset 다운로드 및 읽기 예제
uci_bc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
uci_bc_df = pd.read_csv(uci_bc_url, header=None)

# 데이터 확인
print("\nBreast Cancer Wisconsin (Diagnostic) Dataset:")
print(uci_bc_df.head())
