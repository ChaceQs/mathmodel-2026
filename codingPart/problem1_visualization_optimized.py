"""
共享单车调度优化问题 - 问题1：优化版可视化分析
基于scientific-visualization技能的专业标准优化
- 使用出版级样式和colorblind-safe调色板
- 改进图表专业性和可读性
- 统一视觉风格
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 出版级样式配置 (基于scientific-visualization最佳实践)
# ============================================================================

# 使用Okabe-Ito colorblind-safe调色板
OKABE_ITO = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9', 
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000'
}

# 设置matplotlib出版级样式
plt.rcParams.update({
    # 字体设置
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    
    # 图表样式
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    
    # 其他设置
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# 设置seaborn主题
sns.set_theme(style='ticks', context='paper', font='Microsoft YaHei', font_scale=1.0)
sns.set_palette([OKABE_ITO['orange'], OKABE_ITO['sky_blue'], 
                 OKABE_ITO['green'], OKABE_ITO['vermillion']])

print("="*80)
print("生成优化版可视化图表 (Publication-Quality)")
print("="*80)

# 确保在正确的目录
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"工作目录: {os.getcwd()}\n")

# 读取数据
station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8')
daily_inventory = pd.read_csv('每日初始库存.csv', encoding='utf-8')
weather_data = pd.read_csv('天气数据.csv', encoding='utf-8')
dispatch_records = pd.read_csv('调度记录表.csv', encoding='utf-8')
riding_data = pd.read_csv('日汇总骑行数据.csv', encoding='utf-8')
station_features_df = pd.read_csv('站点特征分析结果.csv', encoding='utf-8')

# 转换日期
riding_data['日期'] = pd.to_datetime(riding_data['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])
dispatch_records['日期'] = pd.to_datetime(dispatch_records['日期'])

print("\n数据加载完成，开始生成图表...")

# ============================================================================
# 图1：站点流量对比（总流入vs总流出）- 优化版
# ============================================================================
print("\n生成图1：站点流量对比...")

fig, ax = plt.subplots(figsize=(12, 6))

station_flow = riding_data.groupby('站点 ID').agg({
    '总流出': 'mean',
    '总流入': 'mean'
}).reset_index()

station_flow = station_flow.merge(station_info[['station_id', '站点名称', '类型']],
                                   left_on='站点 ID', right_on='station_id')

x = np.arange(len(station_flow))
width = 0.35

# 使用colorblind-safe颜色
bars1 = ax.bar(x - width/2, station_flow['总流出'], width, label='平均流出',
               color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, station_flow['总流入'], width, label='平均流入',
               color=OKABE_ITO['sky_blue'], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('站点', fontweight='bold')
ax.set_ylabel('平均车辆数 (辆/天)', fontweight='bold')
ax.set_title('各站点平均流入流出对比', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(station_flow['站点名称'], rotation=45, ha='right')
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')

# 添加数值标签（仅显示整数）
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=7)

sns.despine()
plt.tight_layout()
plt.savefig('图1_站点流量对比_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图1已保存")
plt.close()

# ============================================================================
# 图2：潮汐现象可视化（早晚高峰对比）- 优化版
# ============================================================================
print("生成图2：潮汐现象可视化...")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 计算早晚高峰净流量
riding_data['早高峰净流量'] = riding_data['早高峰流入'] - riding_data['早高峰流出']
riding_data['晚高峰净流量'] = riding_data['晚高峰流入'] - riding_data['晚高峰流出']

tide_data = riding_data.groupby('站点 ID').agg({
    '早高峰净流量': 'mean',
    '晚高峰净流量': 'mean'
}).reset_index()

tide_data = tide_data.merge(station_info[['station_id', '站点名称', '类型']],
                             left_on='站点 ID', right_on='station_id')

x = np.arange(len(tide_data))

# 早高峰 - 使用统一配色方案
colors_morning = [OKABE_ITO['vermillion'] if val < 0 else OKABE_ITO['sky_blue'] 
                  for val in tide_data['早高峰净流量']]
bars1 = axes[0].bar(x, tide_data['早高峰净流量'], color=colors_morning, 
                    alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1.2)
axes[0].set_ylabel('净流量 (辆)', fontweight='bold')
axes[0].set_title('早高峰净流量 (负值=净流出, 正值=净流入)', fontweight='bold', pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(tide_data['站点名称'], rotation=45, ha='right')

# 晚高峰
colors_evening = [OKABE_ITO['vermillion'] if val < 0 else OKABE_ITO['sky_blue'] 
                  for val in tide_data['晚高峰净流量']]
bars2 = axes[1].bar(x, tide_data['晚高峰净流量'], color=colors_evening, 
                    alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.2)
axes[1].set_ylabel('净流量 (辆)', fontweight='bold')
axes[1].set_title('晚高峰净流量 (负值=净流出, 正值=净流入)', fontweight='bold', pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(tide_data['站点名称'], rotation=45, ha='right')

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('图2_潮汐现象可视化_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图2已保存")
plt.close()

# ============================================================================
# 图3：时间序列分析（典型站点）- 优化版
# ============================================================================
print("生成图3：时间序列分析...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 选择4个典型站点
typical_stations = ['S001', 'S005', 'S008', 'S009']
station_names = {
    'S001': '华苑小区 (居民区)',
    'S005': '科技大厦 (商务区)',
    'S008': '地铁站口 (交通枢纽)',
    'S009': '大学城站 (教育区)'
}

for idx, station_id in enumerate(typical_stations):
    ax = axes[idx // 2, idx % 2]
    
    station_data = riding_data[riding_data['站点 ID'] == station_id].sort_values('日期')
    
    # 使用colorblind-safe颜色和不同标记
    ax.plot(station_data['日期'], station_data['总流出'],
            marker='o', label='流出', linewidth=2, markersize=5, 
            color=OKABE_ITO['vermillion'], markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(station_data['日期'], station_data['总流入'],
            marker='s', label='流入', linewidth=2, markersize=5,
            color=OKABE_ITO['sky_blue'], markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_title(station_names[station_id], fontweight='bold', pad=10)
    ax.set_xlabel('日期', fontweight='bold')
    ax.set_ylabel('车辆数 (辆)', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(axis='x', rotation=45)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('图3_时间序列分析_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图3已保存")
plt.close()

# ============================================================================
# 图4：工作日vs周末对比 - 优化版
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 流出对比
weekday_out = weekday_weekend[weekday_weekend['日期类型'] == '工作日'].set_index('站点名称')['总流出']
weekend_out = weekday_weekend[weekday_weekend['日期类型'] == '周末/节假日'].set_index('站点名称')['总流出']

x = np.arange(len(weekday_out))
width = 0.35

axes[0].bar(x - width/2, weekday_out.values, width, label='工作日', 
            color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].bar(x + width/2, weekend_out.values, width, label='周末/节假日', 
            color=OKABE_ITO['green'], alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('站点', fontweight='bold')
axes[0].set_ylabel('平均流出 (辆)', fontweight='bold')
axes[0].set_title('工作日 vs 周末流出对比', fontweight='bold', pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(weekday_out.index, rotation=45, ha='right')
axes[0].legend(frameon=True, fancybox=False, edgecolor='black')

# 流入对比
weekday_in = weekday_weekend[weekday_weekend['日期类型'] == '工作日'].set_index('站点名称')['总流入']
weekend_in = weekday_weekend[weekday_weekend['日期类型'] == '周末/节假日'].set_index('站点名称')['总流入']

axes[1].bar(x - width/2, weekday_in.values, width, label='工作日', 
            color=OKABE_ITO['sky_blue'], alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].bar(x + width/2, weekend_in.values, width, label='周末/节假日', 
            color=OKABE_ITO['orange'], alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('站点', fontweight='bold')
axes[1].set_ylabel('平均流入 (辆)', fontweight='bold')
axes[1].set_title('工作日 vs 周末流入对比', fontweight='bold', pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(weekday_in.index, rotation=45, ha='right')
axes[1].legend(frameon=True, fancybox=False, edgecolor='black')

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('图4_工作日周末对比_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图4已保存")
plt.close()

# ============================================================================
# 图5：天气影响分析 - 优化版
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
               color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, weather_impact['总流入'], width, label='平均流入',
               color=OKABE_ITO['sky_blue'], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('天气类型', fontweight='bold')
ax.set_ylabel('平均车辆数 (辆)', fontweight='bold')
ax.set_title('不同天气条件下的骑行量对比', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(weather_impact['天气类型'])
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

sns.despine()
plt.tight_layout()
plt.savefig('图5_天气影响分析_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图5已保存")
plt.close()

# ============================================================================
# 图6：站点聚类结果可视化 - 优化版
# ============================================================================
print("生成图6：站点聚类结果...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 使用Okabe-Ito调色板的三种颜色
cluster_colors = [OKABE_ITO['vermillion'], OKABE_ITO['sky_blue'], OKABE_ITO['green']]

# 散点图1：平均净流量 vs 潮汐指数
# 标签偏移量：(水平偏移, 垂直偏移)
# 正值向右/上，负值向左/下
# 如需手动调整，修改下面的数值即可
label_offsets = {
    '华苑小区': (0, 10), 
    '锦绣家园': (-3, -10), 
    '阳光新城': (0, -10), 
    '绿城花园': (3, 10),
    '科技大厦': (0, 10), 
    '金融中心': (0, 10), 
    '时代广场': (0, 10),
    '地铁站口': (-18, 0),      # 向左移动，垂直居中
    '大学城站': (12, 10), 
    '商业街口': (0, -16)        # 向下移动更多
}

for cluster_id in sorted(station_features_df['聚类标签'].unique()):
    cluster_data = station_features_df[station_features_df['聚类标签'] == cluster_id]
    axes[0].scatter(cluster_data['平均净流量'], cluster_data['潮汐指数'],
                   s=180, alpha=0.8, c=cluster_colors[cluster_id],
                   label=cluster_data['聚类名称'].iloc[0], 
                   edgecolors='black', linewidth=1.2, zorder=3)
    
    for _, row in cluster_data.iterrows():
        offset = label_offsets.get(row['站点名称'], (0, 10))
        axes[0].annotate(row['站点名称'], (row['平均净流量'], row['潮汐指数']),
                        xytext=offset, textcoords='offset points', fontsize=8,
                        ha='center', va='bottom' if offset[1] > 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                 edgecolor='gray', alpha=0.85, linewidth=0.8))

axes[0].set_xlabel('平均净流量 (辆/天)', fontweight='bold')
axes[0].set_ylabel('潮汐指数', fontweight='bold')
axes[0].set_title('站点聚类结果 (净流量-潮汐指数)', fontweight='bold', pad=10)
axes[0].legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# 散点图2：平均流出 vs 平均流入
label_offsets_2 = {
    '华苑小区': (0, 10), '锦绣家园': (0, -10), '阳光新城': (12, 3), '绿城花园': (-12, -3),
    '科技大厦': (0, 10), '金融中心': (0, 10), '时代广场': (0, 10),
    '地铁站口': (0, 10), '大学城站': (0, 10), '商业街口': (0, 10)
}

for cluster_id in sorted(station_features_df['聚类标签'].unique()):
    cluster_data = station_features_df[station_features_df['聚类标签'] == cluster_id]
    axes[1].scatter(cluster_data['平均流出'], cluster_data['平均流入'],
                   s=180, alpha=0.8, c=cluster_colors[cluster_id],
                   label=cluster_data['聚类名称'].iloc[0], 
                   edgecolors='black', linewidth=1.2, zorder=3)
    
    for _, row in cluster_data.iterrows():
        offset = label_offsets_2.get(row['站点名称'], (0, 10))
        ha = 'center' if offset[0] == 0 else ('right' if offset[0] < 0 else 'left')
        va = 'bottom' if offset[1] > 0 else ('top' if offset[1] < 0 else 'center')
        axes[1].annotate(row['站点名称'], (row['平均流出'], row['平均流入']),
                        xytext=offset, textcoords='offset points', fontsize=8,
                        ha=ha, va=va,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                 edgecolor='gray', alpha=0.85, linewidth=0.8))

# 添加对角线
max_val = max(station_features_df['平均流出'].max(), station_features_df['平均流入'].max())
axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1, label='流入=流出')

axes[1].set_xlabel('平均流出 (辆/天)', fontweight='bold')
axes[1].set_ylabel('平均流入 (辆/天)', fontweight='bold')
axes[1].set_title('站点聚类结果 (流出-流入)', fontweight='bold', pad=10)
axes[1].legend(frameon=True, fancybox=False, edgecolor='black', loc='best')

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('图6_站点聚类结果_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图6已保存")
plt.close()

# ============================================================================
# 图7：关键站点识别 - 优化版
# ============================================================================
print("生成图7：关键站点识别...")

fig, ax = plt.subplots(figsize=(10, 7))

station_features_sorted = station_features_df.sort_values('关键度评分', ascending=True)

# 使用渐变色表示关键度
median_score = station_features_sorted['关键度评分'].median()
colors_key = [OKABE_ITO['vermillion'] if score > median_score
              else OKABE_ITO['sky_blue'] for score in station_features_sorted['关键度评分']]

bars = ax.barh(station_features_sorted['站点名称'],
               station_features_sorted['关键度评分'],
               color=colors_key, alpha=0.85, edgecolor='black', linewidth=0.8)

ax.set_xlabel('关键度评分', fontweight='bold')
ax.set_ylabel('站点', fontweight='bold')
ax.set_title('站点关键度排名 (综合流量、潮汐、调度频次)', fontweight='bold', pad=15)

# 添加数值标签
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f}',
            ha='left', va='center', fontsize=8, fontweight='bold')

# 添加中位线
ax.axvline(x=median_score, color=OKABE_ITO['black'], linestyle='--', 
           linewidth=1.5, label=f'中位数 = {median_score:.1f}', alpha=0.7)
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')

sns.despine()
plt.tight_layout()
plt.savefig('图7_关键站点识别_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图7已保存")
plt.close()

# ============================================================================
# 图8：调度模式分析 - 优化版
# ============================================================================
print("生成图8：调度模式分析...")

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

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 调入调出对比
x = np.arange(len(dispatch_df))
width = 0.35

axes[0].bar(x - width/2, dispatch_df['调入总量'], width, label='调入总量',
            color=OKABE_ITO['sky_blue'], alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].bar(x + width/2, dispatch_df['调出总量'], width, label='调出总量',
            color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('站点', fontweight='bold')
axes[0].set_ylabel('车辆数 (辆)', fontweight='bold')
axes[0].set_title('各站点调度总量对比', fontweight='bold', pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(dispatch_df['站点名称'], rotation=45, ha='right')
axes[0].legend(frameon=True, fancybox=False, edgecolor='black')

# 净调度
colors_net = [OKABE_ITO['vermillion'] if val < 0 else OKABE_ITO['sky_blue'] 
              for val in dispatch_df['净调度']]
bars = axes[1].bar(x, dispatch_df['净调度'], color=colors_net, 
                   alpha=0.85, edgecolor='black', linewidth=0.8)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.2)
axes[1].set_xlabel('站点', fontweight='bold')
axes[1].set_ylabel('净调度量 (辆)', fontweight='bold')
axes[1].set_title('各站点净调度量 (负值=净调出, 正值=净调入)', fontweight='bold', pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(dispatch_df['站点名称'], rotation=45, ha='right')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if abs(height) > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('图8_调度模式分析_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图8已保存")
plt.close()

# ============================================================================
# 图9：特征相关性热力图 - 优化版
# ============================================================================
print("生成图9：特征相关性热力图...")

correlation_features = [
    '平均流出', '平均流入', '平均净流量', '潮汐指数',
    '工作日周末流出差异', '净调度', '关键度评分'
]

corr_matrix = station_features_df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))

# 使用colorblind-safe的diverging colormap (RdBu_r)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=1.5, linecolor='white',
            cbar_kws={"shrink": 0.8, "label": "相关系数"},
            ax=ax, vmin=-1, vmax=1,
            annot_kws={'fontsize': 9, 'fontweight': 'bold'})

ax.set_title('站点特征相关性热力图', fontweight='bold', pad=15, fontsize=12)

# 优化标签
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('图9_特征相关性热力图_优化版.png', dpi=300, bbox_inches='tight')
print("[OK] 图9已保存")
plt.close()

print("\n" + "="*80)
print("所有优化版图表生成完成！")
print("="*80)
print("\n生成的图表文件（优化版）：")
print("  1. 图1_站点流量对比_优化版.png")
print("  2. 图2_潮汐现象可视化_优化版.png")
print("  3. 图3_时间序列分析_优化版.png")
print("  4. 图4_工作日周末对比_优化版.png")
print("  5. 图5_天气影响分析_优化版.png")
print("  6. 图6_站点聚类结果_优化版.png")
print("  7. 图7_关键站点识别_优化版.png")
print("  8. 图8_调度模式分析_优化版.png")
print("  9. 图9_特征相关性热力图_优化版.png")
print("\n优化特点：")
print("  ✓ 使用Okabe-Ito colorblind-safe调色板")
print("  ✓ 出版级字体和样式设置")
print("  ✓ 统一的视觉风格和专业外观")
print("  ✓ 改进的标签可读性和图表清晰度")
print("  ✓ 符合scientific-visualization最佳实践")
