from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=20, step=1)
selector = selector.fit(X, y)

# Lấy danh sách đặc trưng đã chọn
selected_features = X.columns[selector.support_]
print("RFE-selected features:", selected_features)