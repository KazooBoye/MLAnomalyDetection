import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read training and testing datasets
train_df = pd.read_csv('cleanedMulticlass_KddTrain+.csv')
test_df = pd.read_csv('cleanedMulticlass_KddTest+.csv')

# Seperate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Initiate and training KDD model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Perform predictions on Test dataset
y_pred = knn.predict(X_test)

# Print Statistics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
