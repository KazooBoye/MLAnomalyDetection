import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler

# 1. Load preprocessed datasets (đã chuẩn hóa từ trước)
train_df = pd.read_csv("cleanedMulticlass_KddTrain+.csv")
test_df = pd.read_csv("cleanedMulticlass_KddTest+.csv")

# 2. Tách nhãn và đặc trưng
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 3. Đảm bảo tập test có các cột giống tập train
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

# 4. Áp dụng SMOTE (hoặc fallback) để cân bằng tập train
label_counts = Counter(y_train)
min_class_size = min(label_counts.values())

if min_class_size >= 2:
    safe_k = max(1, min(min_class_size - 1, 5))
    print(f"✔️ Sử dụng SMOTE với k_neighbors = {safe_k} (Lớp nhỏ nhất có {min_class_size} mẫu)")
    sm = SMOTE(random_state=42, k_neighbors=safe_k)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    print(f"⚠️ Có lớp chỉ có 1 mẫu. Dùng RandomOverSampler thay vì SMOTE")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# 5. Huấn luyện mô hình Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

# 6. Dự đoán và đánh giá trên tập test
y_pred = rf.predict(X_test)

print("=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
