"""
共享单车调度优化问题 - 问题1：数据特征分析与站点分类
作者：数学建模团队
日期：2026-05-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("共享单车调度优化 - 问题1：数据特征分析与站点分类")
print("="*80)

# ============================================================================
# 第一部分：数据读取与初步检查
# ============================================================================
print("\n【第一部分：数据读取与质量评估】\n")

# 读取所有数据文件
station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8')
daily_inventory = pd.read_csv('每日初始库存.csv', encoding='utf-8')
weather_data = pd.read_csv('天气数据.csv', encoding='utf-8')
dispatch_records = pd.read_csv('调度记录表.csv', encoding='utf-8')
expected_inventory = pd.read_csv('站点期望库存.csv', encoding='utf-8')
station_distance = pd.read_csv('站点距离.csv', encoding='utf-8')
riding_data = pd.read_csv('日汇总骑行数据.csv', encoding='utf-8')

print("1. 数据文件读取成功")
print(f"   - 站点基础信息: {station_info.shape}")
print(f"   - 每日初始库存: {daily_inventory.shape}")
print(f"   - 天气数据: {weather_data.shape}")
print(f"   - 调度记录表: {dispatch_records.shape}")
print(f"   - 站点期望库存: {expected_inventory.shape}")
print(f"   - 站点距离: {station_distance.shape}")
print(f"   - 日汇总骑行数据: {riding_data.shape}")

# ============================================================================
# 第二部分：数据质量评估
# ============================================================================
print("\n2. 数据质量评估")

# 2.1 缺失值检查
print("\n2.1 缺失值检查：")
datasets = {
    '站点基础信息': station_info,
    '每日初始库存': daily_inventory,
    '天气数据': weather_data,
    '调度记录表': dispatch_records,
    '站点期望库存': expected_inventory,
    '日汇总骑行数据': riding_data
}

missing_summary = []
for name, df in datasets.items():
    missing_count = df.isnull().sum().sum()
    missing_summary.append({
        '数据表': name,
        '总记录数': len(df),
        '缺失值数量': missing_count,
        '缺失率': f"{missing_count / (len(df) * len(df.columns)) * 100:.2f}%"
    })

missing_df = pd.DataFrame(missing_summary)
print(missing_df.to_string(index=False))

# 2.2 数据类型检查
print("\n2.2 数据类型检查：")
print("   站点基础信息数据类型：")
print(station_info.dtypes)

# 2.3 数值范围检查
print("\n2.3 数值范围合理性检查：")
print("   站点容量统计：")
print(station_info[['站点名称', '容量']].describe())

print("\n   每日初始库存统计（前5个站点）：")
inventory_cols = [col for col in daily_inventory.columns if col.startswith('S')]
print(daily_inventory[inventory_cols[:5]].describe())

# ============================================================================
# 第三部分：数据逻辑一致性验证
# ============================================================================
print("\n\n【第二部分：数据逻辑一致性验证】\n")

# 3.1 库存-骑行-调度一致性验证
print("3.1 库存变化与骑行、调度数据的一致性验证：")

# 转换日期格式
daily_inventory['日期'] = pd.to_datetime(daily_inventory['日期'])
riding_data['日期'] = pd.to_datetime(riding_data['日期'])
dispatch_records['日期'] = pd.to_datetime(dispatch_records['日期'])

consistency_results = []

for i in range(len(daily_inventory) - 1):
    date = daily_inventory.loc[i, '日期']
    next_date = daily_inventory.loc[i + 1, '日期']

    for station_id in inventory_cols:
        # 当日初始库存
        initial_inventory = daily_inventory.loc[i, station_id]
        # 次日初始库存
        next_initial_inventory = daily_inventory.loc[i + 1, station_id]

        # 当日骑行数据
        riding_day = riding_data[riding_data['日期'] == date]
        riding_station = riding_day[riding_day['站点 ID'] == station_id]

        if len(riding_station) > 0:
            outflow = riding_station['总流出'].values[0]
            inflow = riding_station['总流入'].values[0]
            net_flow = inflow - outflow
        else:
            net_flow = 0

        # 当日调度数据
        dispatch_day = dispatch_records[dispatch_records['日期'] == date]
        dispatch_in = dispatch_day[dispatch_day['调入站点'] == station_id]['调度车辆数'].sum()
        dispatch_out = dispatch_day[dispatch_day['调出站点'] == station_id]['调度车辆数'].sum()
        net_dispatch = dispatch_in - dispatch_out

        # 理论次日库存 = 当日初始库存 + 净流量 + 净调度
        theoretical_next = initial_inventory + net_flow + net_dispatch
        actual_next = next_initial_inventory

        difference = abs(theoretical_next - actual_next)

        if difference > 0:
            consistency_results.append({
                '日期': date.strftime('%Y-%m-%d'),
                '站点': station_id,
                '初始库存': initial_inventory,
                '净流量': net_flow,
                '净调度': net_dispatch,
                '理论次日库存': theoretical_next,
                '实际次日库存': actual_next,
                '差异': difference
            })

if len(consistency_results) > 0:
    consistency_df = pd.DataFrame(consistency_results)
    print(f"   发现 {len(consistency_df)} 条不一致记录")
    print(f"   平均差异: {consistency_df['差异'].mean():.2f} 辆")
    print(f"   最大差异: {consistency_df['差异'].max():.2f} 辆")
    print("\n   前10条不一致记录：")
    print(consistency_df.head(10).to_string(index=False))
else:
    print("   [OK] 所有数据逻辑一致，未发现矛盾")

# 3.2 容量约束检查
print("\n3.2 站点容量约束检查：")
capacity_violations = []

for i, row in daily_inventory.iterrows():
    date = row['日期']
    for station_id in inventory_cols:
        inventory = row[station_id]
        capacity = station_info[station_info['station_id'] == station_id]['容量'].values[0]

        if inventory > capacity:
            capacity_violations.append({
                '日期': date.strftime('%Y-%m-%d'),
                '站点': station_id,
                '库存': inventory,
                '容量': capacity,
                '超出': inventory - capacity
            })

if len(capacity_violations) > 0:
    print(f"   [WARNING] 发现 {len(capacity_violations)} 条容量超限记录")
    print(pd.DataFrame(capacity_violations).head(10).to_string(index=False))
else:
    print("   [OK] 所有库存均未超过站点容量")

# 3.3 调度时间窗口检查
print("\n3.3 调度时间窗口检查（应在00:00-05:00）：")
dispatch_records['开始时间_dt'] = pd.to_datetime(dispatch_records['开始时间'], format='%H:%M:%S').dt.hour
dispatch_records['结束时间_dt'] = pd.to_datetime(dispatch_records['结束时间'], format='%H:%M:%S').dt.hour

time_violations = dispatch_records[
    (dispatch_records['开始时间_dt'] >= 5) |
    (dispatch_records['结束时间_dt'] >= 5)
]

if len(time_violations) > 0:
    print(f"   [WARNING] 发现 {len(time_violations)} 条时间窗口违规记录")
else:
    print("   [OK] 所有调度均在规定时间窗口内（00:00-05:00）")

# ============================================================================
# 第四部分：时空特征分析
# ============================================================================
print("\n\n【第三部分：时空特征深度分析】\n")

# 4.1 站点流量特征分析
print("4.1 站点流量特征统计：\n")

# 按站点汇总流量数据
station_flow_summary = riding_data.groupby('站点 ID').agg({
    '总流出': ['mean', 'std', 'max', 'min'],
    '总流入': ['mean', 'std', 'max', 'min'],
    '净流量': ['mean', 'std', 'max', 'min']
}).round(2)

station_flow_summary.columns = ['_'.join(col) for col in station_flow_summary.columns]
print(station_flow_summary)

# 4.2 潮汐现象分析
print("\n\n4.2 潮汐现象强度分析：\n")

# 计算潮汐指数：早高峰净流出 vs 晚高峰净流入
riding_data['早高峰净流出'] = riding_data['早高峰流出'] - riding_data['早高峰流入']
riding_data['晚高峰净流入'] = riding_data['晚高峰流入'] - riding_data['晚高峰流出']
riding_data['潮汐指数'] = abs(riding_data['早高峰净流出']) + abs(riding_data['晚高峰净流入'])

tide_summary = riding_data.groupby('站点 ID').agg({
    '早高峰净流出': 'mean',
    '晚高峰净流入': 'mean',
    '潮汐指数': 'mean'
}).round(2)

# 合并站点类型信息
tide_summary = tide_summary.merge(
    station_info[['station_id', '站点名称', '类型']],
    left_index=True,
    right_on='station_id'
).set_index('station_id')

print(tide_summary.sort_values('潮汐指数', ascending=False))

# 4.3 工作日vs周末对比
print("\n\n4.3 工作日与周末流量对比：\n")

# 合并天气数据获取星期信息（确保日期格式一致）
weather_data['日期'] = pd.to_datetime(weather_data['日期'])
riding_data = riding_data.merge(weather_data[['日期', '星期', '是否节假日']], on='日期')
riding_data['日期类型'] = riding_data['是否节假日'].apply(lambda x: '周末/节假日' if x == 1 else '工作日')

weekday_weekend = riding_data.groupby(['站点 ID', '日期类型']).agg({
    '总流出': 'mean',
    '总流入': 'mean',
    '净流量': 'mean'
}).round(2)

print(weekday_weekend)

# 4.4 天气影响分析
print("\n\n4.4 天气对骑行量的影响分析：\n")

# 确保天气数据日期格式一致
weather_data['日期'] = pd.to_datetime(weather_data['日期'])
riding_data = riding_data.merge(weather_data[['日期', '天气类型', '天气系数']], on='日期', how='left', suffixes=('', '_y'))
riding_data = riding_data.loc[:, ~riding_data.columns.str.endswith('_y')]

weather_impact = riding_data.groupby('天气类型').agg({
    '总流出': 'mean',
    '总流入': 'mean',
    '净流量': 'mean'
}).round(2)

print(weather_impact)

# ============================================================================
# 第五部分：站点分类与关键特征提取
# ============================================================================
print("\n\n【第四部分：站点分类与关键特征提取】\n")

# 5.1 构建站点特征矩阵
print("5.1 构建站点特征矩阵...\n")

# 为每个站点计算综合特征
station_features = []

for station_id in inventory_cols:
    # 基础信息
    station_row = station_info[station_info['station_id'] == station_id].iloc[0]

    # 流量特征
    station_riding = riding_data[riding_data['站点 ID'] == station_id]

    # 潮汐特征
    tide_data = tide_summary.loc[station_id]

    # 工作日周末差异
    weekday_data = riding_data[(riding_data['站点 ID'] == station_id) & (riding_data['日期类型'] == '工作日')]
    weekend_data = riding_data[(riding_data['站点 ID'] == station_id) & (riding_data['日期类型'] == '周末/节假日')]

    features = {
        'station_id': station_id,
        '站点名称': station_row['站点名称'],
        '站点类型': station_row['类型'],
        '容量': station_row['容量'],
        '需求等级': station_row['需求等级'],

        # 流量特征
        '平均流出': station_riding['总流出'].mean(),
        '平均流入': station_riding['总流入'].mean(),
        '平均净流量': station_riding['净流量'].mean(),
        '流出标准差': station_riding['总流出'].std(),
        '流入标准差': station_riding['总流入'].std(),

        # 潮汐特征
        '早高峰净流出': tide_data['早高峰净流出'],
        '晚高峰净流入': tide_data['晚高峰净流入'],
        '潮汐指数': tide_data['潮汐指数'],

        # 工作日周末差异
        '工作日平均流出': weekday_data['总流出'].mean() if len(weekday_data) > 0 else 0,
        '周末平均流出': weekend_data['总流出'].mean() if len(weekend_data) > 0 else 0,
        '工作日周末流出差异': weekday_data['总流出'].mean() - weekend_data['总流出'].mean() if len(weekday_data) > 0 and len(weekend_data) > 0 else 0,

        # 调度特征
        '调入次数': len(dispatch_records[dispatch_records['调入站点'] == station_id]),
        '调出次数': len(dispatch_records[dispatch_records['调出站点'] == station_id]),
        '净调度': len(dispatch_records[dispatch_records['调入站点'] == station_id]) - len(dispatch_records[dispatch_records['调出站点'] == station_id])
    }

    station_features.append(features)

station_features_df = pd.DataFrame(station_features)
print("站点特征矩阵（前5行）：")
print(station_features_df.head().to_string(index=False))

# 5.2 K-means聚类分析
print("\n\n5.2 基于K-means的站点聚类分析...\n")

# 选择用于聚类的数值特征
clustering_features = [
    '平均流出', '平均流入', '平均净流量',
    '潮汐指数', '工作日周末流出差异', '净调度'
]

X = station_features_df[clustering_features].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法则确定最优聚类数
inertias = []
silhouette_scores = []
K_range = range(2, 6)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

print("不同聚类数的评估指标：")
for k, inertia, silhouette in zip(K_range, inertias, silhouette_scores):
    print(f"   K={k}: 惯性={inertia:.2f}, 轮廓系数={silhouette:.3f}")

# 选择K=3进行聚类（根据站点类型：居民区、商务区、交通/教育）
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
station_features_df['聚类标签'] = kmeans.fit_predict(X_scaled)

print(f"\n采用K={optimal_k}进行聚类，结果如下：\n")
print(station_features_df[['station_id', '站点名称', '站点类型', '聚类标签', '潮汐指数', '平均净流量']].to_string(index=False))

# 5.3 聚类结果分析
print("\n\n5.3 各聚类的特征统计：\n")

cluster_summary = station_features_df.groupby('聚类标签')[clustering_features].mean().round(2)
print(cluster_summary)

# 为聚类命名
cluster_names = {}
for cluster_id in range(optimal_k):
    cluster_data = station_features_df[station_features_df['聚类标签'] == cluster_id]
    avg_net_flow = cluster_data['平均净流量'].mean()
    avg_tide = cluster_data['潮汐指数'].mean()

    if avg_net_flow > 10:
        cluster_names[cluster_id] = "需求型站点（商务区）"
    elif avg_net_flow < -5:
        cluster_names[cluster_id] = "供给型站点（居民区）"
    else:
        cluster_names[cluster_id] = "平衡型站点（交通枢纽）"

station_features_df['聚类名称'] = station_features_df['聚类标签'].map(cluster_names)

print("\n\n聚类命名结果：")
for cluster_id, name in cluster_names.items():
    stations = station_features_df[station_features_df['聚类标签'] == cluster_id]['站点名称'].tolist()
    print(f"   聚类{cluster_id} - {name}: {', '.join(stations)}")

# 5.4 识别关键站点
print("\n\n5.4 关键站点识别：\n")

# 综合评分：考虑流量、潮汐强度、调度频次
station_features_df['关键度评分'] = (
    station_features_df['平均流出'] * 0.3 +
    station_features_df['平均流入'] * 0.3 +
    station_features_df['潮汐指数'] * 0.2 +
    abs(station_features_df['净调度']) * 0.2
)

station_features_df = station_features_df.sort_values('关键度评分', ascending=False)

print("关键站点排名（按关键度评分）：")
print(station_features_df[['station_id', '站点名称', '站点类型', '聚类名称', '关键度评分']].to_string(index=False))

# 保存特征数据
station_features_df.to_csv('站点特征分析结果.csv', index=False, encoding='utf-8-sig')
print("\n[OK] 站点特征分析结果已保存至 '站点特征分析结果.csv'")

print("\n" + "="*80)
print("数据分析完成！开始生成可视化图表...")
print("="*80)
