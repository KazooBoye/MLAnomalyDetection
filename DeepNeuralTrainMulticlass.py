import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

# === Load & prepare data ===
train_df = pd.read_csv("cleaned5Grouped_KddTrain+.csv")
test_df = pd.read_csv("cleaned5Grouped_KddTest+.csv")

# Handle NaN labels in test data if any
if 'label' in test_df.columns:
    nan_mask = test_df['label'].isnull()
    if nan_mask.any():
        print(f"Found and dropping {nan_mask.sum()} rows with NaN labels in test data.")
        test_df = test_df.dropna(subset=['label'])

# Prepare training data
X_train = train_df.drop(['label'], axis=1)
y_train = train_df['label'].values

# Prepare test data  
X_test = test_df.drop(['label'], axis=1)
y_test = test_df['label'].values

# Ensure test set has same columns as training set
train_columns = X_train.columns
missing_cols = set(train_columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[train_columns]

# === Build a deep Neural Network Classifier ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))

# Deep hidden layers
hidden1 = Dense(128, activation='relu')(input_layer)
hidden2 = Dense(64, activation='relu')(hidden1)
hidden3 = Dense(32, activation='relu')(hidden2)

# Bottleneck
bottleneck = Dense(16, activation='relu')(hidden3)

# Output layer for 5-class classification
output = Dense(5, activation='softmax')(bottleneck)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# === Train ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stop])

# === Save the trained model for future use ===
model.save("multiclass_classifier_model.h5")
print("Model saved as 'multiclass_classifier_model.h5'")

# === Make predictions ===
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# === Evaluation ===
print("\n=== 5-Class Neural Network Classification Report ===")
class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
print(classification_report(y_test, y_pred, target_names=class_names))

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# === Plot training history ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
