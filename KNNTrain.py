import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load preprocessed data
train_df = pd.read_csv('cleanedBinary_KddTrain+.csv')
test_df = pd.read_csv('cleanedBinary_KddTest+.csv')

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Remove rows with missing labels
nan_mask = y_test.isnull()
if nan_mask.any():
    print(f"Found and dropping {nan_mask.sum()} rows with NaN labels in test data.")
    y_test = y_test.dropna()
    X_test = X_test.loc[y_test.index]

# Synchronize columns between training and testing sets
train_columns = X_train.columns
missing_cols = set(train_columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[train_columns]

# === Find the optimal k ===
print("Finding the optimal k value...")

accuracies = []
k_range = range(1, 50)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)

# Select the optimal k
optimal_k = k_range[accuracies.index(max(accuracies))]
print(f"Optimal k value: {optimal_k} with accuracy: {max(accuracies):.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='-')
plt.title('Accuracy by Number of Neighbors (k)')
plt.xlabel('Value of k')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# === Train with optimal k ===
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)

# Evaluation
print("=== Classification Results with Optimal k ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
