# 导入必要的库
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# 显示中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载
order_details = pd.read_csv(r"C:\Users\ROG\Desktop\机器学习实验\实验五\data\meal_order_detail.csv")
order_info = pd.read_csv(r"C:\Users\ROG\Desktop\机器学习实验\实验五\data\meal_order_info.csv")
dishes_info = pd.read_csv(r"C:\Users\ROG\Desktop\机器学习实验\实验五\data\meal_dishes_detail.csv")

# 数据预览
print("Order Details Preview:\n", order_details.head())
print("Order Info Preview:\n", order_info.head())
print("Dishes Info Preview:\n", dishes_info.head())

# 2. 数据清洗与关联
order_details['counts'] = pd.to_numeric(order_details['counts'], errors='coerce').fillna(0).astype(int)
order_info = order_info.rename(columns={"info_id": "order_id"})
completed_orders = order_info[order_info['order_status'] == 1]  # 筛选订单状态为 1 的记录
completed_orders['use_start_time'] = pd.to_datetime(completed_orders['use_start_time'])
completed_orders['date'] = completed_orders['use_start_time'].dt.date

# 合并数据
merged_orders = pd.merge(order_details, completed_orders, on='order_id', how='inner')
merged_data = pd.merge(merged_orders, dishes_info, left_on='dishes_id', right_on='id', how='left')

# 确保统一的菜品名称列
if 'dishes_name_x' in merged_data.columns or 'dishes_name_y' in merged_data.columns:
    merged_data['dishes_name'] = merged_data['dishes_name_x']  # 或根据需要改为 dishes_name_y
else:
    merged_data['dishes_name'] = merged_data['dishes_name']

# 3. 每日用餐人数与营业额分析
daily_data = merged_data.groupby('date').agg(daily_consumers=('number_consumers', 'sum'),
                                             daily_revenue=('amounts', 'sum')).reset_index()
print("每日用餐人数与营业额:\n", daily_data)

# 绘制每日用餐人数与营业额的折线图
plt.figure(figsize=(12, 6))
plt.plot(daily_data['date'], daily_data['daily_revenue'], label="营业额", color="orange", linestyle="-")
plt.plot(daily_data['date'], daily_data['daily_consumers'], label="用餐人数", color="blue", linestyle="--")
plt.xlabel("日期")
plt.ylabel("数量 / 金额")
plt.title("每日用餐人数与营业额")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. 热销度分析
# 统计菜品的销售数量
dishes_sales = merged_data.groupby('dishes_name').agg(sales_quantity=('counts', 'sum')).reset_index()

# 计算热销度评分
Q_min, Q_max = dishes_sales['sales_quantity'].min(), dishes_sales['sales_quantity'].max()
dishes_sales['popularity_score'] = (dishes_sales['sales_quantity'] - Q_min) / (Q_max - Q_min)

# 提取热销度评分前 10 的菜品
top10_dishes = dishes_sales.sort_values(by='popularity_score', ascending=False).head(10)
print("热销度评分前 10 的菜品:\n", top10_dishes)

# 绘制热销度前 10 菜品的柱状图
plt.figure(figsize=(10, 6))
plt.bar(top10_dishes['dishes_name'], top10_dishes['sales_quantity'], color="skyblue")
plt.xlabel("菜品名称")
plt.ylabel("销售数量")
plt.title("热销度前 10 菜品")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. 使用 Apriori 算法进行关联分析
basket = merged_data.groupby(['order_id', 'dishes_name'])['counts'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: x > 0)

# 挖掘频繁项集
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print(f"生成的频繁项集数量: {len(frequent_itemsets)}")

if not frequent_itemsets.empty:  # 确保频繁项集非空
    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(f"生成的关联规则数量: {len(rules)}")

    # 模型评价
    avg_lift = rules['lift'].mean()
    print(f"规则的平均提升度: {avg_lift:.2f}")

    # 可视化关联规则
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', marker='o')
    plt.colorbar(label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Support-Confidence-Lift")
    plt.tight_layout()
    plt.show()

    # 输出提升度高的规则建议
    recommend_rules = rules[rules['lift'] > 2]
    print("提升度大于 2 的规则:\n", recommend_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print("未找到足够的频繁项集，无法生成关联规则。")
