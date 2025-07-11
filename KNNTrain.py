import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read training and testing datasets
train_df = pd.read_csv('cleaned5Grouped_KddTrain+_SMOTE.csv')
test_df = pd.read_csv('cleaned5Grouped_KddTest+.csv')

# Seperate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Drop rows with NaN in y_test and corresponding rows in X_test
nan_mask = y_test.isnull()
if nan_mask.any():
    print(f"Found and dropping {nan_mask.sum()} rows with NaN labels in test data.")
    y_test = y_test.dropna()
    X_test = X_test.loc[y_test.index]


# Ensure test set has the same columns as training set
train_columns = X_train.columns
missing_cols = set(train_columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[train_columns]

# Initiate and training KDD model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Perform predictions on Test dataset
y_pred = knn.predict(X_test)

# Print Statistics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
