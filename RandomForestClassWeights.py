import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load datasets (đã được tiền xử lý và chuẩn hóa)
train_df = pd.read_csv("cleanedMulticlass_KddTrain+.csv")
test_df = pd.read_csv("cleanedMulticlass_KddTest+.csv")

# 2. Split features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 3. Ensure test set has same columns as training set
train_columns = X_train.columns
missing_cols = set(train_columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[train_columns]

# 4. Train Random Forest with class weights
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # 🔥 Đây là phần quan trọng
)
rf.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = rf.predict(X_test)

print("=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
