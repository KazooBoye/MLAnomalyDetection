import pandas as pd

df = pd.read_csv('cleaned5Grouped_KddTest+.csv')
print(df['label'].isnull().sum())  # Kết quả mong muốn là 0
