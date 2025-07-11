import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load tập train đã xử lý để làm chuẩn cho các cột
train_df = pd.read_csv("cleaned5Grouped_KddTrain+.csv")
expected_columns = set(train_df.columns) - {"label"}

# Tên cột ban đầu
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_level']

# Đọc dữ liệu gốc
df = pd.read_csv('./KDDTest+.txt', names=columns)

# Xử lý cơ bản
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['duration'] >= 0]

# One-hot encoding cho 3 cột phân loại
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

# Gộp nhãn tấn công về 5 nhóm
label_map = {
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'normal': 'Normal'
}

df['label'] = df['label'].map(label_map)
df = df[df['label'].notnull()]  # Bỏ các dòng không nằm trong 5 nhóm

# Tách dữ liệu và nhãn
X = df.drop(['label', 'difficulty_level'], axis=1)
y = df['label']

# Chuẩn hóa
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Đồng bộ các cột với tập train
current_columns = set(X.columns)
missing_cols = expected_columns - current_columns
for col in missing_cols:
    X[col] = 0  # Thêm cột bị thiếu với giá trị 0

# Sắp xếp lại thứ tự cột theo tập train
X = X[sorted(expected_columns)]

# Gán nhãn số cho 5 lớp
label_groups = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
y_encoded = y.map(label_groups)

# Gộp và lưu
df_cleaned = X.copy()
df_cleaned['label'] = y_encoded
df_cleaned.to_csv("cleaned5Grouped_KddTest+.csv", index=False)

print("Xử lý hoàn tất.")
print("Nhóm nhãn:", label_groups)
