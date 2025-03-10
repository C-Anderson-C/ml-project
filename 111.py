import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import rcParams

rcParams['font.family'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def load_and_analyze_data(file_path):
    data = pd.read_csv(file_path)

    data.columns = data.columns.str.strip()

    print("数据集的基本信息：")
    print(data.info())

    print("\n数据的描述性统计：")
    print(data.describe())

    print("\n缺失值统计：")
    print(data.isnull().sum())

    target_variable = 'incomplete_flag'
    print("\n目标变量的分布：")
    sns.histplot(data[target_variable], kde=True)
    plt.title(f'{target_variable} 分布')
    plt.show()

    numeric_data = data.select_dtypes(include=[np.number])

    print("\n特征之间的相关性：")
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('特征相关性热力图')
    plt.show()

    return data



def preprocess_data(data):
    target_variable = 'incomplete_flag'
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    X = X.loc[:, X.isnull().mean() < 1]

    X['start_time_DI'] = pd.to_datetime(X['start_time_DI'], format='%d/%m/%y', errors='coerce')
    X['last_event_DI'] = pd.to_datetime(X['last_event_DI'], format='%d/%m/%y', errors='coerce')

    time_columns = ['start_time_DI', 'last_event_DI']
    for col in time_columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    numeric_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    numeric_imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_columns])

    X_numeric = pd.DataFrame(X[numeric_columns], columns=numeric_columns)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    X = pd.concat([X_numeric, X_encoded_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    print("\n数据预处理后：")
    print(f"训练集大小：{X_train.shape[0]} 样本")
    print(f"验证集大小：{X_val.shape[0]} 样本")
    print(f"测试集大小：{X_test.shape[0]} 样本")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val, y_val_pred)

        y_test_pred = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_test_pred)

        results.append({
            'Model': model_name,
            'Val MSE': mse_val, 'Val RMSE': rmse_val, 'Val R2': r2_val,
            'Test MSE': mse_test, 'Test RMSE': rmse_test, 'Test R2': r2_test
        })

        print(f"\n{model_name} 在验证集上的表现：")
        print(f"MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")
        print(f"{model_name} 在测试集上的表现：")
        print(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

    results_df = pd.DataFrame(results)
    print("\n模型评估结果：")
    print(results_df)


def main():
    file_path = r"C:\Users\ROG\Desktop\机器学习实验\实验二\吴宇捷 9109222145\HXPC13_DI_v3_11-13-2019_preprocessed.csv"
    data = load_and_analyze_data(file_path)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)

    build_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()