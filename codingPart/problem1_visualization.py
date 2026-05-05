"""
共享单车调度优化问题 - 问题1：可视化分析
生成各类特征分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

# 统一配色方案
COLORS = {
    'primary': '#2E5090',      # 深蓝
    'secondary': '#E67E22',    # 橙色
    'accent': '#27AE60',       # 绿色
    'warning': '#E74C3C',      # 红色
    'neutral': '#2C3E50'       # 深灰
}

CATEGORICAL_COLORS = [
    '#2E5090', '#E67E22', '#27AE60', '#E74C3C',
    '#9B59B6', '#16A085', '#F39C12', '#34495E'
]

sns.set_palette(CATEGORICAL_COLORS)
sns.set_style("whitegrid")

# 设置中文字体（必须在 seaborn 之后，否则会被 sns.set_style 覆盖）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("生成可视化图表")
print("="*80)

# 读取数据
station_info = pd.read_csv('附件数据/站点基础信息.csv', encoding='utf-8')
daily_inventory = pd.read_csv('附件数据/每日初始库存.csv', encoding='utf-8')
weather_data = pd.read_csv('附件数据/天气数据.csv', encoding='utf-8')
dispatch_records = pd.read_csv('附件数据/调度记录表.csv', encoding='utf-8')
riding_data = pd.read_csv('附件数据/日汇总骑行数据.csv', encoding='utf-8')
station_features_df = pd.read_csv('站点特征分析结果.csv', encoding='utf-8')

# 转换日期
riding_data['日期'] = pd.to_datetime(riding_data['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])

# ============================================================================
# 图1：站点流量对比（总流入vs总流出）
# ============================================================================
print("\n生成图1：站点流量对比...")

fig, ax = plt.subplots(figsize=(14, 8))

station_flow = riding_data.groupby('站点 ID').agg({
    '总流出': 'mean',
    '总流入': 'mean'
}).reset_index()

station_flow = station_flow.merge(station_info[['station_id', '站点名称', '类型']],
                                   left_on='站点 ID', right_on='station_id')

x = np.arange(len(station_flow))
width = 0.35

bars1 = ax.bar(x - width/2, station_flow['总流出'], width, label='平均流出',
               color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x + width/2, station_flow['总流入'], width, label='平均流入',
               color='#4ECDC4', alpha=0.8)

ax.set_xlabel('站点', fontsize=12, fontweight='bold')
ax.set_ylabel('平均车辆数（辆/天）', fontsize=12, fontweight='bold')
ax.set_title('各站点平均流入流出对比', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(station_flow['站点名称'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('图1_站点流量对比.png', dpi=300, bbox_inches='tight')
print("[OK] 图1已保存")
plt.close()

# ============================================================================
# 图2：潮汐现象可视化（早晚高峰对比）
# ============================================================================
print("生成图2：潮汐现象可视化...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 计算早晚高峰净流量
riding_data['早高峰净流量'] = riding_data['早高峰流入'] - riding_data['早高峰流出']
riding_data['晚高峰净流量'] = riding_data['晚高峰流入'] - riding_data['晚高峰流出']

tide_data = riding_data.groupby('站点 ID').agg({
    '早高峰净流量': 'mean',
    '晚高峰净流量': 'mean'
}).reset_index()

tide_data = tide_data.merge(station_info[['station_id', '站点名称', '类型']],
                             left_on='站点 ID', right_on='station_id')

# 早高峰
x = np.arange(len(tide_data))
colors_morning = ['#FF6B6B' if val < 0 else '#4ECDC4' for val in tide_data['早高峰净流量']]
bars1 = axes[0].bar(x, tide_data['早高峰净流量'], color=colors_morning, alpha=0.8)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_ylabel('净流量（辆）', fontsize=11, fontweight='bold')
axes[0].set_title('早高峰净流量（负值=流出>流入，正值=流入>流出）', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(tide_data['站点名称'], rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# 晚高峰
colors_evening = ['#FF6B6B' if val < 0 else '#4ECDC4' for val in tide_data['晚高峰净流量']]
bars2 = axes[1].bar(x, tide_data['晚高峰净流量'], color=colors_evening, alpha=0.8)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_ylabel('净流量（辆）', fontsize=11, fontweight='bold')
axes[1].set_title('晚高峰净流量（负值=流出>流入，正值=流入>流出）', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(tide_data['站点名称'], rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图2_潮汐现象可视化.png', dpi=300, bbox_inches='tight')
print("[OK] 图2已保存")
plt.close()

# ============================================================================
# 图3：时间序列分析（典型站点）
# ============================================================================
print("生成图3：时间序列分析...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 选择4个典型站点
typical_stations = ['S001', 'S005', 'S008', 'S009']
station_names = {
    'S001': '华苑小区（居民区）',
    'S005': '科技大厦（商务区）',
    'S008': '地铁站口（交通枢纽）',
    'S009': '大学城站（教育区）'
}

for idx, station_id in enumerate(typical_stations):
    ax = axes[idx // 2, idx % 2]

    station_data = riding_data[riding_data['站点 ID'] == station_id].sort_values('日期')

    ax.plot(station_data['日期'], station_data['总流出'],
            marker='o', label='流出', linewidth=2, markersize=6, color='#FF6B6B')
    ax.plot(station_data['日期'], station_data['总流入'],
            marker='s', label='流入', linewidth=2, markersize=6, color='#4ECDC4')

    ax.set_title(station_names[station_id], fontsize=12, fontweight='bold')
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('车辆数（辆）', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('图3_时间序列分析.png', dpi=300, bbox_inches='tight')
print("[OK] 图3已保存")
plt.close()

# ============================================================================
# 图4：工作日vs周末对比
# ============================================================================
print("生成图4：工作日vs周末对比...")

riding_data = riding_data.merge(weather_data[['日期', '是否节假日']], on='日期', how='left')
riding_data['日期类型'] = riding_data['是否节假日'].apply(lambda x: '周末/节假日' if x == 1 else '工作日')

weekday_weekend = riding_data.groupby(['站点 ID', '日期类型']).agg({
    '总流出': 'mean',
    '总流入': 'mean'
}).reset_index()

weekday_weekend = weekday_weekend.merge(station_info[['station_id', '站点名称']],
                                         left_on='站点 ID', right_on='station_id')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 流出对比
weekday_out = weekday_weekend[weekday_weekend['日期类型'] == '工作日'].set_index('站点名称')['总流出']
weekend_out = weekday_weekend[weekday_weekend['日期类型'] == '周末/节假日'].set_index('站点名称')['总流出']

x = np.arange(len(weekday_out))
width = 0.35

axes[0].bar(x - width/2, weekday_out.values, width, label='工作日', color='#FF6B6B', alpha=0.8)
axes[0].bar(x + width/2, weekend_out.values, width, label='周末/节假日', color='#95E1D3', alpha=0.8)
axes[0].set_xlabel('站点', fontsize=11, fontweight='bold')
axes[0].set_ylabel('平均流出（辆）', fontsize=11, fontweight='bold')
axes[0].set_title('工作日vs周末流出对比', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(weekday_out.index, rotation=45, ha='right')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# 流入对比
weekday_in = weekday_weekend[weekday_weekend['日期类型'] == '工作日'].set_index('站点名称')['总流入']
weekend_in = weekday_weekend[weekday_weekend['日期类型'] == '周末/节假日'].set_index('站点名称')['总流入']

axes[1].bar(x - width/2, weekday_in.values, width, label='工作日', color='#4ECDC4', alpha=0.8)
axes[1].bar(x + width/2, weekend_in.values, width, label='周末/节假日', color='#F38181', alpha=0.8)
axes[1].set_xlabel('站点', fontsize=11, fontweight='bold')
axes[1].set_ylabel('平均流入（辆）', fontsize=11, fontweight='bold')
axes[1].set_title('工作日vs周末流入对比', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(weekday_in.index, rotation=45, ha='right')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图4_工作日周末对比.png', dpi=300, bbox_inches='tight')
print("[OK] 图4已保存")
plt.close()

# ============================================================================
# 图5：天气影响分析
# ============================================================================
print("生成图5：天气影响分析...")

riding_data = riding_data.merge(weather_data[['日期', '天气类型']], on='日期', how='left')

weather_impact = riding_data.groupby('天气类型').agg({
    '总流出': 'mean',
    '总流入': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(weather_impact))
width = 0.35

bars1 = ax.bar(x - width/2, weather_impact['总流出'], width, label='平均流出',
               color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x + width/2, weather_impact['总流入'], width, label='平均流入',
               color='#4ECDC4', alpha=0.8)

ax.set_xlabel('天气类型', fontsize=12, fontweight='bold')
ax.set_ylabel('平均车辆数（辆）', fontsize=12, fontweight='bold')
ax.set_title('不同天气条件下的骑行量对比', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(weather_impact['天气类型'], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('图5_天气影响分析.png', dpi=300, bbox_inches='tight')
print("[OK] 图5已保存")
plt.close()

# ============================================================================
# 图6：站点聚类结果可视化
# ============================================================================
print("生成图6：站点聚类结果...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 散点图：平均净流量 vs 潮汐指数
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

# 手动定义标签位置偏移，避免重叠
label_offsets = {
    '华苑小区': (0, 8),
    '锦绣家园': (-3, -8),
    '阳光新城': (0, -8),
    '绿城花园': (3, 8),
    '科技大厦': (0, 8),
    '金融中心': (0, 8),
    '时代广场': (0, 8),
    '地铁站口': (-10, -8),      # 向左下偏移，避免与商业街口重叠
    '大学城站': (10, 8),         # 向右上偏移，避免与商业街口重叠
    '商业街口': (0, -10)         # 向下偏移更多，避免与地铁站口重叠
}

for cluster_id in station_features_df['聚类标签'].unique():
    cluster_data = station_features_df[station_features_df['聚类标签'] == cluster_id]
    axes[0].scatter(cluster_data['平均净流量'], cluster_data['潮汐指数'],
                   s=200, alpha=0.7, c=colors[cluster_id],
                   label=cluster_data['聚类名称'].iloc[0], edgecolors='black', linewidth=1.5)

    # 添加站点标签（使用偏移避免重叠）
    for _, row in cluster_data.iterrows():
        offset = label_offsets.get(row['站点名称'], (0, 8))
        axes[0].annotate(row['站点名称'],
                        (row['平均净流量'], row['潮汐指数']),
                        xytext=offset, textcoords='offset points',
                        fontsize=9, ha='center', va='bottom' if offset[1] > 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))

axes[0].set_xlabel('平均净流量（辆/天）', fontsize=11, fontweight='bold')
axes[0].set_ylabel('潮汐指数', fontsize=11, fontweight='bold')
axes[0].set_title('站点聚类结果（净流量-潮汐指数）', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

# 散点图：平均流出 vs 平均流入
# 手动定义标签位置偏移，避免重叠
label_offsets_2 = {
    '华苑小区': (0, 8),
    '锦绣家园': (0, -8),
    '阳光新城': (10, 3),        # 向右偏移，避免与绿城花园重叠
    '绿城花园': (-10, -3),      # 向左下偏移，避免与阳光新城重叠
    '科技大厦': (0, 8),
    '金融中心': (0, 8),
    '时代广场': (0, 8),
    '地铁站口': (0, 8),
    '大学城站': (0, 8),
    '商业街口': (0, 8)
}

for cluster_id in station_features_df['聚类标签'].unique():
    cluster_data = station_features_df[station_features_df['聚类标签'] == cluster_id]
    axes[1].scatter(cluster_data['平均流出'], cluster_data['平均流入'],
                   s=200, alpha=0.7, c=colors[cluster_id],
                   label=cluster_data['聚类名称'].iloc[0], edgecolors='black', linewidth=1.5)

    # 添加站点标签（使用偏移避免重叠）
    for _, row in cluster_data.iterrows():
        offset = label_offsets_2.get(row['站点名称'], (0, 8))
        ha = 'center' if offset[0] == 0 else ('right' if offset[0] < 0 else 'left')
        va = 'bottom' if offset[1] > 0 else ('top' if offset[1] < 0 else 'center')
        axes[1].annotate(row['站点名称'],
                        (row['平均流出'], row['平均流入']),
                        xytext=offset, textcoords='offset points',
                        fontsize=9, ha=ha, va=va,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))

# 添加对角线（流入=流出）
max_val = max(station_features_df['平均流出'].max(), station_features_df['平均流入'].max())
axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

axes[1].set_xlabel('平均流出（辆/天）', fontsize=11, fontweight='bold')
axes[1].set_ylabel('平均流入（辆/天）', fontsize=11, fontweight='bold')
axes[1].set_title('站点聚类结果（流出-流入）', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('图6_站点聚类结果.png', dpi=300, bbox_inches='tight')
print("[OK] 图6已保存")
plt.close()

# ============================================================================
# 图7：关键站点识别
# ============================================================================
print("生成图7：关键站点识别...")

fig, ax = plt.subplots(figsize=(14, 8))

station_features_sorted = station_features_df.sort_values('关键度评分', ascending=True)

colors_key = ['#FF6B6B' if score > station_features_sorted['关键度评分'].median()
              else '#4ECDC4' for score in station_features_sorted['关键度评分']]

bars = ax.barh(station_features_sorted['站点名称'],
               station_features_sorted['关键度评分'],
               color=colors_key, alpha=0.8, edgecolor='black', linewidth=1)

ax.set_xlabel('关键度评分', fontsize=12, fontweight='bold')
ax.set_ylabel('站点', fontsize=12, fontweight='bold')
ax.set_title('站点关键度排名（综合流量、潮汐、调度频次）', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# 添加数值标签
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

# 添加中位线
median_score = station_features_sorted['关键度评分'].median()
ax.axvline(x=median_score, color='red', linestyle='--', linewidth=2, label=f'中位数={median_score:.1f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('图7_关键站点识别.png', dpi=300, bbox_inches='tight')
print("[OK] 图7已保存")
plt.close()

# ============================================================================
# 图8：调度模式分析
# ============================================================================
print("生成图8：调度模式分析...")

dispatch_records['日期'] = pd.to_datetime(dispatch_records['日期'])

# 统计每个站点的调入调出
dispatch_summary = []
for station_id in station_info['station_id']:
    dispatch_in = dispatch_records[dispatch_records['调入站点'] == station_id]['调度车辆数'].sum()
    dispatch_out = dispatch_records[dispatch_records['调出站点'] == station_id]['调度车辆数'].sum()

    station_name = station_info[station_info['station_id'] == station_id]['站点名称'].values[0]

    dispatch_summary.append({
        'station_id': station_id,
        '站点名称': station_name,
        '调入总量': dispatch_in,
        '调出总量': dispatch_out,
        '净调度': dispatch_in - dispatch_out
    })

dispatch_df = pd.DataFrame(dispatch_summary)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 调入调出对比
x = np.arange(len(dispatch_df))
width = 0.35

axes[0].bar(x - width/2, dispatch_df['调入总量'], width, label='调入总量',
            color='#4ECDC4', alpha=0.8)
axes[0].bar(x + width/2, dispatch_df['调出总量'], width, label='调出总量',
            color='#FF6B6B', alpha=0.8)
axes[0].set_xlabel('站点', fontsize=11, fontweight='bold')
axes[0].set_ylabel('车辆数（辆）', fontsize=11, fontweight='bold')
axes[0].set_title('各站点调度总量对比', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(dispatch_df['站点名称'], rotation=45, ha='right')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# 净调度
colors_net = ['#FF6B6B' if val < 0 else '#4ECDC4' for val in dispatch_df['净调度']]
bars = axes[1].bar(x, dispatch_df['净调度'], color=colors_net, alpha=0.8, edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_xlabel('站点', fontsize=11, fontweight='bold')
axes[1].set_ylabel('净调度量（辆）', fontsize=11, fontweight='bold')
axes[1].set_title('各站点净调度量（负值=调出>调入，正值=调入>调出）', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(dispatch_df['站点名称'], rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('图8_调度模式分析.png', dpi=300, bbox_inches='tight')
print("[OK] 图8已保存")
plt.close()

# ============================================================================
# 图9：相关性热力图
# ============================================================================
print("生成图9：特征相关性热力图...")

correlation_features = [
    '平均流出', '平均流入', '平均净流量', '潮汐指数',
    '工作日周末流出差异', '净调度', '关键度评分'
]

corr_matrix = station_features_df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('站点特征相关性热力图', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('图9_特征相关性热力图.png', dpi=300, bbox_inches='tight')
print("[OK] 图9已保存")
plt.close()

print("\n" + "="*80)
print("所有图表生成完成！")
print("="*80)
print("\n生成的图表文件：")
print("  1. 图1_站点流量对比.png")
print("  2. 图2_潮汐现象可视化.png")
print("  3. 图3_时间序列分析.png")
print("  4. 图4_工作日周末对比.png")
print("  5. 图5_天气影响分析.png")
print("  6. 图6_站点聚类结果.png")
print("  7. 图7_关键站点识别.png")
print("  8. 图8_调度模式分析.png")
print("  9. 图9_特征相关性热力图.png")
