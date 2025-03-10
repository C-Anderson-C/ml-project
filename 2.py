import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_path = r"C:\Users\ROG\Desktop\机器学习实验\实验一\HXPC13_DI_v3_11-13-2019_preprocessed.csv"
data = pd.read_csv(data_path, low_memory=False)

print("数据集概览：")
print(data.info())

y = data['certified']
X = data.drop(columns=['certified', 'course_id', 'userid_DI', 'start_time_DI', 'last_event_DI'], errors='ignore')

print("是否存在空值：", X.isnull().sum().sum())

X = X.fillna(0)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
