import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

file_path = 'C:\\Users\\ROG\\Desktop\\机器学习实验\\实验一\\HXPC13_DI_v3_11-13-2019.csv'
data = pd.read_csv(file_path)

def preprocess_data(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].apply(lambda x: re.sub(r'\s+', '', str(x)) if pd.notna(x) else x)
        df[column] = df[column].apply(lambda x: x.replace('.', '/').replace('-', '/') if pd.notna(x) else x)
    return df

data = preprocess_data(data)

def parse_date_column(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = df[column_name].apply(lambda x: x.strftime('%y/%m/%d') if pd.notna(x) else '00/00/00')
    return df

date_columns = ['start_time_DI', 'last_event_DI']
for column in date_columns:
    data = parse_date_column(data, column)

def handle_missing_values(df):

    for column in df.select_dtypes(include=['number']).columns:
        df[column] = df[column].fillna(df[column].median())

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df

data = handle_missing_values(data)

print(data.info())
print(data.describe())

plt.figure(figsize=(10, 6))
sns.histplot(data['grade'], kde=True)
plt.title('Grade Distribution')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='gender', data=data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

output_file_path = 'C:\\Users\\ROG\\Desktop\\机器学习实验\\实验一\\HXPC13_DI_v3_11-13-2019_preprocessed.csv'
data.to_csv(output_file_path, index=False)
