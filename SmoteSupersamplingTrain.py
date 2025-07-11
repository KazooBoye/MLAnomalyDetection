import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load tập dữ liệu đã tiền xử lý và nhóm nhãn
df = pd.read_csv("cleaned5Grouped_KddTrain+.csv")

# Tách feature và label
X = df.drop('label', axis=1)
y = df['label']

# Kiểm tra phân phối ban đầu
print("Initial class distribution:")
print(Counter(y))

# Chọn số k phù hợp theo lớp nhỏ nhất (SMOTE yêu cầu >= k+1 samples)
min_class_count = y.value_counts().min()
k_neighbors = min(min_class_count - 1, 5) if min_class_count > 1 else 1

# Áp dụng SMOTE để cân bằng các lớp
print(f"\nUsing SMOTE with k_neighbors = {k_neighbors}")
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Gộp lại thành DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['label'] = y_resampled

# Kiểm tra lại phân phối sau SMOTE
print("\n Distribution after SMOTE:")
print(Counter(y_resampled))

# Lưu kết quả ra file mới
df_resampled.to_csv("cleaned5Grouped_KddTrain+_SMOTE.csv", index=False)
print("\n Saved: cleaned5Grouped_KddTrain+_SMOTE.csv")
