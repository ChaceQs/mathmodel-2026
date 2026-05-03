"""
问题二：共享单车调度优化
两阶段建模方案：统计预测 + 数学优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from pulp import *
import warnings
warnings.filterwarnings('ignore')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("问题二：共享单车调度优化建模")
print("="*80)

# ============================================================================
# 第一部分：数据加载与预处理
# ============================================================================
print("\n[第一部分] 数据加载与预处理")
print("-"*80)

# 读取数据
station_info = pd.read_csv('站点基础信息.csv')
station_features = pd.read_csv('站点特征分析结果.csv')
expected_inventory = pd.read_csv('站点期望库存.csv')
initial_inventory = pd.read_csv('每日初始库存.csv')
distance_matrix = pd.read_csv('站点距离.csv', index_col=0)
weather_data = pd.read_csv('天气数据.csv')
riding_data = pd.read_csv('日汇总骑行数据.csv')

print(f"[OK] 数据加载完成")
print(f"  - 站点数量: {len(station_info)}")
print(f"  - 数据天数: {len(initial_inventory)}")

# 数据预处理
initial_inventory['日期'] = pd.to_datetime(initial_inventory['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])
riding_data['日期'] = pd.to_datetime(riding_data['日期'])

# 将初始库存从宽格式转为长格式
initial_inventory_long = initial_inventory.melt(
    id_vars=['日期'],
    var_name='站点ID',
    value_name='初始库存'
)

# 重命名骑行数据列名以匹配
riding_data_renamed = riding_data.rename(columns={
    '站点 ID': '站点ID',
    '总流出': '流出量',
    '总流入': '流入量'
})

# 合并数据
data = initial_inventory_long.merge(weather_data, on='日期', how='left')
data = data.merge(riding_data_renamed[['日期', '站点ID', '流入量', '流出量']],
                  on=['日期', '站点ID'],
                  how='left')

print(f"[OK] 数据合并完成，总记录数: {len(data)}")

# ============================================================================
# 第二部分：特征工程
# ============================================================================
print("\n[第二部分] 特征工程")
print("-"*80)

# 从站点特征表获取关键特征
station_features_dict = station_features.set_index('station_id').to_dict('index')

# 为每条记录添加站点特征
data['容量'] = data['站点ID'].map(lambda x: station_features_dict.get(x, {}).get('容量', 0))
data['关键度评分'] = data['站点ID'].map(lambda x: station_features_dict.get(x, {}).get('关键度评分', 0))
data['潮汐指数'] = data['站点ID'].map(lambda x: station_features_dict.get(x, {}).get('潮汐指数', 0))
data['聚类标签'] = data['站点ID'].map(lambda x: station_features_dict.get(x, {}).get('聚类标签', 0))

# 时间特征
data['星期几'] = data['日期'].dt.dayofweek
data['是否工作日'] = data['星期几'].apply(lambda x: 1 if x < 5 else 0)
data['月份'] = data['日期'].dt.month
data['日'] = data['日期'].dt.day

# 周期性编码
data['星期_sin'] = np.sin(2 * np.pi * data['星期几'] / 7)
data['星期_cos'] = np.cos(2 * np.pi * data['星期几'] / 7)

# 天气特征编码
weather_mapping = {'晴': 0, '多云': 1, '阴': 2, '小雨': 3, '雨': 4}
data['天气编码'] = data['天气'].map(weather_mapping)

# 期望库存（根据工作日/周末）
expected_dict = expected_inventory.set_index('站点ID').to_dict('index')
def get_expected_inventory(row):
    station_id = row['站点ID']
    is_workday = row['是否工作日']
    if station_id in expected_dict:
        if is_workday == 1:
            return expected_dict[station_id].get('工作日期望库存', 0)
        else:
            return expected_dict[station_id].get('周末期望库存', 0)
    return 0

data['期望库存'] = data.apply(get_expected_inventory, axis=1)

# 库存偏差
data['库存偏差'] = data['初始库存'] - data['期望库存']
data['库存偏差_abs'] = data['库存偏差'].abs()

# 容量利用率
data['容量利用率'] = data['初始库存'] / data['容量']

# 滞后特征（前一天的库存）
data = data.sort_values(['站点ID', '日期'])
data['前一天库存'] = data.groupby('站点ID')['初始库存'].shift(1)
data['前一天流出'] = data.groupby('站点ID')['流出量'].shift(1)
data['前一天流入'] = data.groupby('站点ID')['流入量'].shift(1)

# 滚动统计特征（7日均值）
data['流出_7日均值'] = data.groupby('站点ID')['流出量'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
data['流入_7日均值'] = data.groupby('站点ID')['流入量'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# 删除缺失值
data = data.dropna()

print(f"[OK] 特征工程完成")
print(f"  - 特征数量: {len(data.columns)}")
print(f"  - 有效记录数: {len(data)}")

# ============================================================================
# 第三部分：XGBoost预测模型（预测期望库存偏差）
# ============================================================================
print("\n[第三部分] XGBoost预测模型训练")
print("-"*80)

# 选择特征
feature_cols = [
    '初始库存', '容量', '关键度评分', '潮汐指数', '聚类标签',
    '是否工作日', '星期_sin', '星期_cos', '天气编码', '是否节假日',
    '容量利用率', '前一天库存', '前一天流出', '前一天流入',
    '流出_7日均值', '流入_7日均值'
]

X = data[feature_cols]
y = data['库存偏差_abs']  # 预测库存偏差的绝对值

# 数据分割（时间序列分割）
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# 训练XGBoost模型
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=5,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

print("开始训练XGBoost模型...")
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

# 模型评估
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n[OK] 模型训练完成")
print(f"训练集性能:")
print(f"  - R² = {train_r2:.4f}")
print(f"  - RMSE = {train_rmse:.4f}")
print(f"  - MAE = {train_mae:.4f}")
print(f"测试集性能:")
print(f"  - R² = {test_r2:.4f}")
print(f"  - RMSE = {test_rmse:.4f}")
print(f"  - MAE = {test_mae:.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': xgb_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n特征重要性 Top 5:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['特征']}: {row['重要性']:.4f}")

# ============================================================================
# 第四部分：MILP调度优化模型
# ============================================================================
print("\n[第四部分] MILP调度优化模型")
print("-"*80)

# 选择最后一天的数据进行调度优化
last_date = data['日期'].max()
last_day_data = data[data['日期'] == last_date].copy()

print(f"优化日期: {last_date.strftime('%Y-%m-%d')}")
print(f"站点数量: {len(last_day_data)}")

# 准备优化模型的参数
stations = last_day_data['站点ID'].tolist()
N = len(stations)

# 初始库存
I0 = dict(zip(last_day_data['站点ID'], last_day_data['初始库存']))

# 期望库存
I_star = dict(zip(last_day_data['站点ID'], last_day_data['期望库存']))

# 容量
C = dict(zip(last_day_data['站点ID'], last_day_data['容量']))

# 权重（关键度评分归一化）
weights_raw = dict(zip(last_day_data['站点ID'], last_day_data['关键度评分']))
max_weight = max(weights_raw.values())
weights = {k: v/max_weight for k, v in weights_raw.items()}

# 距离矩阵
dist = distance_matrix.loc[stations, stations].to_dict()

# 卡车参数
M = 3  # 卡车数量
trucks = [f'T{i+1}' for i in range(M)]
Q = {truck: 15 for truck in trucks}  # 每辆卡车容量15辆

# 时间参数
v = 30  # 平均速度 30 km/h
t_load = 10  # 装车时间 10分钟
t_unload = 10  # 卸车时间 10分钟
T_max = 300  # 时间窗口 300分钟（5小时）

# 计算时间矩阵
time_matrix = {}
for i in stations:
    time_matrix[i] = {}
    for j in stations:
        if i != j:
            travel_time = (dist[i][j] / v) * 60  # 转换为分钟
            time_matrix[i][j] = travel_time + t_load + t_unload
        else:
            time_matrix[i][j] = 0

# 成本参数
c_fuel = 2.0  # 燃油成本 2元/km
c_time = 0.5  # 时间成本 0.5元/分钟

# 权重系数
alpha = 0.7  # 库存偏差权重
beta = 0.2   # 成本权重
gamma = 0.1  # 次数权重

print(f"\n模型参数:")
print(f"  - 站点数: {N}")
print(f"  - 卡车数: {M}")
print(f"  - 卡车容量: {Q['T1']} 辆")
print(f"  - 时间窗口: {T_max} 分钟")
print(f"  - 权重系数: α={alpha}, β={beta}, γ={gamma}")

# 创建优化模型
print("\n开始构建MILP模型...")
prob = LpProblem("Bike_Scheduling_Optimization", LpMinimize)

# 决策变量
x = {}  # 调度量
y = {}  # 是否调度
for k in trucks:
    for i in stations:
        for j in stations:
            if i != j:
                x[k,i,j] = LpVariable(f"x_{k}_{i}_{j}", lowBound=0, cat='Integer')
                y[k,i,j] = LpVariable(f"y_{k}_{i}_{j}", cat='Binary')

# 辅助变量
I = {i: LpVariable(f"I_{i}", lowBound=0, cat='Integer') for i in stations}
xi_plus = {i: LpVariable(f"xi_plus_{i}", lowBound=0) for i in stations}
xi_minus = {i: LpVariable(f"xi_minus_{i}", lowBound=0) for i in stations}

# 目标函数
Z1 = lpSum([weights[i] * (xi_plus[i] + xi_minus[i]) for i in stations])
Z2 = lpSum([y[k,i,j] * (c_fuel * dist[i][j] + c_time * time_matrix[i][j])
            for k in trucks for i in stations for j in stations if i != j])
Z3 = lpSum([y[k,i,j] for k in trucks for i in stations for j in stations if i != j])

# 归一化处理
Z2_normalized = Z2 / 1000  # 成本归一化
Z3_normalized = Z3 / 10    # 次数归一化

prob += alpha * Z1 + beta * Z2_normalized + gamma * Z3_normalized

# 约束条件
print("添加约束条件...")

# 约束1: 库存守恒
for i in stations:
    prob += I[i] == I0[i] + lpSum([x[k,j,i] for k in trucks for j in stations if j != i]) \
                          - lpSum([x[k,i,j] for k in trucks for j in stations if j != i])

# 约束2: 容量约束
for i in stations:
    prob += I[i] <= C[i]

# 约束3: 偏差定义
for i in stations:
    prob += I[i] - I_star[i] == xi_plus[i] - xi_minus[i]

# 约束4: 卡车载重约束
for k in trucks:
    for i in stations:
        for j in stations:
            if i != j:
                prob += x[k,i,j] <= Q[k] * y[k,i,j]

# 约束5: 调度逻辑约束
for k in trucks:
    for i in stations:
        for j in stations:
            if i != j:
                prob += x[k,i,j] >= y[k,i,j]

# 约束6: 时间窗口约束（简化版）
for k in trucks:
    prob += lpSum([y[k,i,j] * time_matrix[i][j]
                   for i in stations for j in stations if i != j]) <= T_max

# 约束7: 调度可行性约束
for i in stations:
    prob += lpSum([x[k,i,j] for k in trucks for j in stations if j != i]) <= I0[i]

print(f"[OK] 模型构建完成")
print(f"  - 决策变量数: {len(x) + len(y) + len(I) + len(xi_plus) + len(xi_minus)}")
print(f"  - 约束条件数: {len(prob.constraints)}")

# 求解模型
print("\n开始求解MILP模型...")
print("(这可能需要几分钟时间...)")

solver = PULP_CBC_CMD(msg=0, timeLimit=300)  # 5分钟时间限制
prob.solve(solver)

print(f"\n[OK] 求解完成")
print(f"求解状态: {LpStatus[prob.status]}")

if prob.status == 1:  # 最优解
    print(f"目标函数值: {value(prob.objective):.4f}")

    # ============================================================================
    # 第五部分：结果分析与可视化
    # ============================================================================
    print("\n[第五部分] 结果分析与可视化")
    print("-"*80)

    # 提取调度方案
    schedule = []
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j and value(y[k,i,j]) == 1:
                    schedule.append({
                        '卡车': k,
                        '起点': i,
                        '终点': j,
                        '调度量': int(value(x[k,i,j])),
                        '距离': dist[i][j],
                        '时间': time_matrix[i][j],
                        '成本': c_fuel * dist[i][j] + c_time * time_matrix[i][j]
                    })

    schedule_df = pd.DataFrame(schedule)

    if len(schedule_df) > 0:
        print(f"\n调度方案汇总:")
        print(f"  - 总调度次数: {len(schedule_df)}")
        print(f"  - 总调度量: {schedule_df['调度量'].sum()} 辆")
        print(f"  - 总距离: {schedule_df['距离'].sum():.2f} km")
        print(f"  - 总时间: {schedule_df['时间'].sum():.2f} 分钟")
        print(f"  - 总成本: {schedule_df['成本'].sum():.2f} 元")

        print(f"\n详细调度方案:")
        for idx, row in schedule_df.iterrows():
            print(f"  {row['卡车']}: {row['起点']} → {row['终点']}, "
                  f"调度 {row['调度量']} 辆, "
                  f"距离 {row['距离']:.1f}km, "
                  f"时间 {row['时间']:.1f}分钟")
    else:
        print("\n无需调度（所有站点库存已达到期望值）")

    # 库存对比
    inventory_comparison = []
    for i in stations:
        inventory_comparison.append({
            '站点': i,
            '初始库存': I0[i],
            '期望库存': I_star[i],
            '调度后库存': int(value(I[i])),
            '初始偏差': abs(I0[i] - I_star[i]),
            '调度后偏差': abs(value(I[i]) - I_star[i])
        })

    inventory_df = pd.DataFrame(inventory_comparison)

    print(f"\n库存优化效果:")
    print(f"  - 调度前总偏差: {inventory_df['初始偏差'].sum():.0f} 辆")
    print(f"  - 调度后总偏差: {inventory_df['调度后偏差'].sum():.2f} 辆")
    print(f"  - 偏差降低: {(1 - inventory_df['调度后偏差'].sum() / inventory_df['初始偏差'].sum()) * 100:.1f}%")

    # 保存结果
    if len(schedule_df) > 0:
        schedule_df.to_csv('问题2_调度方案.csv', index=False, encoding='utf-8-sig')
    inventory_df.to_csv('问题2_库存对比.csv', index=False, encoding='utf-8-sig')

    print(f"\n[OK] 结果已保存:")
    if len(schedule_df) > 0:
        print(f"  - 问题2_调度方案.csv")
    print(f"  - 问题2_库存对比.csv")

    # 可视化
    print("\n生成可视化图表...")

    # 图1: 库存对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 库存对比柱状图
    ax1 = axes[0, 0]
    x_pos = np.arange(len(inventory_df))
    width = 0.25
    ax1.bar(x_pos - width, inventory_df['初始库存'], width, label='初始库存', alpha=0.8)
    ax1.bar(x_pos, inventory_df['期望库存'], width, label='期望库存', alpha=0.8)
    ax1.bar(x_pos + width, inventory_df['调度后库存'], width, label='调度后库存', alpha=0.8)
    ax1.set_xlabel('站点', fontsize=12)
    ax1.set_ylabel('库存量（辆）', fontsize=12)
    ax1.set_title('图1: 站点库存对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(inventory_df['站点'], rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 子图2: 偏差对比
    ax2 = axes[0, 1]
    x_pos = np.arange(len(inventory_df))
    width = 0.35
    ax2.bar(x_pos - width/2, inventory_df['初始偏差'], width, label='调度前偏差', alpha=0.8, color='coral')
    ax2.bar(x_pos + width/2, inventory_df['调度后偏差'], width, label='调度后偏差', alpha=0.8, color='lightgreen')
    ax2.set_xlabel('站点', fontsize=12)
    ax2.set_ylabel('偏差（辆）', fontsize=12)
    ax2.set_title('图2: 库存偏差对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(inventory_df['站点'], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 子图3: 调度网络图
    if len(schedule_df) > 0:
        ax3 = axes[1, 0]
        # 简化的网络图
        for idx, row in schedule_df.iterrows():
            start_idx = stations.index(row['起点'])
            end_idx = stations.index(row['终点'])
            ax3.arrow(start_idx, 0, end_idx - start_idx, 0,
                     head_width=0.3, head_length=0.5, fc='blue', ec='blue', alpha=0.6,
                     length_includes_head=True)
            ax3.text((start_idx + end_idx) / 2, 0.2, f"{row['调度量']}辆",
                    ha='center', fontsize=9)
        ax3.set_xlim(-1, len(stations))
        ax3.set_ylim(-1, 1)
        ax3.set_xticks(range(len(stations)))
        ax3.set_xticklabels(stations, rotation=45)
        ax3.set_yticks([])
        ax3.set_title('图3: 调度网络示意图', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, '无调度操作', ha='center', va='center', fontsize=16)
        ax3.set_title('图3: 调度网络示意图', fontsize=14, fontweight='bold')
        ax3.axis('off')

    # 子图4: 特征重要性
    ax4 = axes[1, 1]
    top_features = feature_importance.head(10)
    ax4.barh(range(len(top_features)), top_features['重要性'], alpha=0.8, color='steelblue')
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['特征'])
    ax4.set_xlabel('重要性', fontsize=12)
    ax4.set_title('图4: XGBoost特征重要性 (Top 10)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('问题2_综合分析图.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存: 问题2_综合分析图.png")

else:
    print(f"[ERROR] 模型求解失败，状态: {LpStatus[prob.status]}")

print("\n" + "="*80)
print("问题二建模完成")
print("="*80)
