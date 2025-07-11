import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers

# Load data
train_df = pd.read_csv("cleaned5Grouped_KddTrain+.csv")
test_df = pd.read_csv("cleaned5Grouped_KddTest+.csv")

# Tách đặc trưng và nhãn
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# Chuyển nhãn thành Binary: Normal = 0, Attack = 1
y_train_binary = (y_train != 0).astype(int)
y_test_binary = (y_test != 0).astype(int)

# Huấn luyện Autoencoder **CHỈ TRÊN NORMAL** data
X_train_normal = X_train[y_train_binary == 0]

# Kiến trúc Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 32  # số chiều ẩn

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Huấn luyện
autoencoder.fit(X_train_normal, X_train_normal,
                epochs=30,
                batch_size=128,
                shuffle=True,
                validation_split=0.1,
                verbose=1)

# === Dự đoán trên tập Test ===
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Ngưỡng phát hiện anomaly
threshold = np.percentile(mse, 95)
print(f"Threshold (95% quantile) = {threshold:.6f}")

# Dự đoán: 1 = anomaly (attack), 0 = normal
y_pred = (mse > threshold).astype(int)

# === Đánh giá ===
print("\n=== Autoencoder Binary Classification Report ===")
print(classification_report(y_test_binary, y_pred))

auc_score = roc_auc_score(y_test_binary, mse)
print(f"AUC Score: {auc_score:.4f}")
