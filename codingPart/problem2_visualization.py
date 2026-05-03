"""
问题二：共享单车调度优化 - 可视化部分
功能：读取模型结果，生成所有可视化图表

输入文件（由 problem2_model.py 生成）：
  - 问题2_调度方案.csv
  - 问题2_库存对比.csv
  - 问题2_成本效益分析.csv
  - 站点基础信息.csv

输出文件：
  - 问题2_图1_站点库存对比.png
  - 问题2_图2_缺货积压对比.png
  - 问题2_图3_成本结构对比.png
  - 问题2_图4_调度网络示意图.png
  - 问题2_图5_各站点偏差改善.png
  - 问题2_图6_卡车工作量分布.png
  - 问题2_图7_综合性能评估.png

使用方法：
  python problem2_visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("问题二：可视化图表生成")
print("="*80)

# ============================================================================
# 配置与数据加载
# ============================================================================
print("\n[配置] 设置绘图参数...")

# 统一配色方案
COLORS = {
    'primary': '#2E5090',
    'secondary': '#E67E22',
    'accent': '#27AE60',
    'warning': '#E74C3C',
    'neutral': '#2C3E50'
}

# 为了兼容旧代码，创建OKABE_ITO映射
OKABE_ITO = {
    'blue': '#2E5090',
    'orange': '#E67E22',
    'green': '#27AE60',
    'vermillion': '#E74C3C',
    'sky_blue': '#5B7FC7',
    'purple': '#9B59B6',
    'yellow': '#F39C12'
}

print("\n[数据] 读取模型结果...")

schedule_df = pd.read_csv('问题2_调度方案.csv', encoding='utf-8-sig')
inventory_df = pd.read_csv('问题2_库存对比.csv', encoding='utf-8-sig')
cost_df = pd.read_csv('问题2_成本效益分析.csv', encoding='utf-8-sig')
station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8-sig')

print(f"[OK] 数据加载完成")
print(f"  - 调度方案: {len(schedule_df)} 条记录")
print(f"  - 库存对比: {len(inventory_df)} 个站点")
print(f"  - 成本效益: {len(cost_df)} 个指标")

T_max = 300

truck_colors = {
    'T1': OKABE_ITO['blue'],
    'T2': OKABE_ITO['orange'],
    'T3': OKABE_ITO['green']
}

initial_shortage_cost = cost_df[cost_df['指标'] == '调度前缺货成本(元)']['数值'].values[0]
initial_excess_cost = cost_df[cost_df['指标'] == '调度前积压成本(元)']['数值'].values[0]
initial_total_cost = initial_shortage_cost + initial_excess_cost

final_shortage_cost = cost_df[cost_df['指标'] == '调度后缺货成本(元)']['数值'].values[0]
final_excess_cost = cost_df[cost_df['指标'] == '调度后积压成本(元)']['数值'].values[0]
transport_cost_total = cost_df[cost_df['指标'] == '运输成本(元)']['数值'].values[0]
final_total_cost = final_shortage_cost + final_excess_cost + transport_cost_total

satisfaction_rate_after = float(cost_df[cost_df['指标'] == '调度后需求满足率(%)']['数值'].values[0])

# ========================================================================
# 图1: 站点库存对比
# ========================================================================
print("\n[生成] 图1: 站点库存对比...")

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(inventory_df))
width = 0.25

bars1 = ax.bar(x_pos - width, inventory_df['初始库存'], width, 
               label='初始库存', color=OKABE_ITO['blue'], alpha=0.85, 
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos, inventory_df['期望库存'], width, 
               label='期望库存', color=OKABE_ITO['orange'], alpha=0.85, 
               edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x_pos + width, inventory_df['调度后库存'], width, 
               label='调度后库存', color=OKABE_ITO['green'], alpha=0.85, 
               edgecolor='black', linewidth=0.5)

ax.set_xlabel('站点', fontweight='bold')
ax.set_ylabel('库存量（辆）', fontweight='bold')
ax.set_title('图1: 站点库存对比', fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(inventory_df['站点'], rotation=45, ha='right')
ax.legend(frameon=True, fancybox=False, edgecolor='black')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=7)

sns.despine()
plt.tight_layout()
plt.savefig('问题2_图1_站点库存对比.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图1_站点库存对比.png")
plt.close()

# ========================================================================
# 图2: 缺货与积压对比
# ========================================================================
print("[生成] 图2: 缺货与积压对比...")

fig, ax = plt.subplots(figsize=(8, 6))
categories = ['调度前', '调度后']
shortage_values = [inventory_df['调度前缺货'].sum(), inventory_df['调度后缺货'].sum()]
excess_values = [inventory_df['调度前积压'].sum(), inventory_df['调度后积压'].sum()]

x_cat = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x_cat - width/2, shortage_values, width, label='缺货量', 
               color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_cat + width/2, excess_values, width, label='积压量', 
               color=OKABE_ITO['sky_blue'], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_ylabel('数量（辆）', fontweight='bold')
ax.set_title('图2: 缺货与积压对比', fontweight='bold', pad=15)
ax.set_xticks(x_cat)
ax.set_xticklabels(categories)
ax.legend(frameon=True, fancybox=False, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

sns.despine()
plt.tight_layout()
plt.savefig('问题2_图2_缺货积压对比.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图2_缺货积压对比.png")
plt.close()

# ========================================================================
# 图3: 成本结构对比
# ========================================================================
print("[生成] 图3: 成本结构对比...")

fig, ax = plt.subplots(figsize=(8, 6))
cost_categories = ['调度前', '调度后']
shortage_costs = [initial_shortage_cost, final_shortage_cost]
excess_costs = [initial_excess_cost, final_excess_cost]
transport_costs = [0, transport_cost_total]

x_cat = np.arange(len(cost_categories))
width = 0.6

p1 = ax.bar(x_cat, shortage_costs, width, label='缺货成本', 
            color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
p2 = ax.bar(x_cat, excess_costs, width, bottom=shortage_costs, label='积压成本', 
            color=OKABE_ITO['sky_blue'], alpha=0.85, edgecolor='black', linewidth=0.5)
bottom_sum = [shortage_costs[i] + excess_costs[i] for i in range(len(cost_categories))]
p3 = ax.bar(x_cat, transport_costs, width, bottom=bottom_sum, label='运输成本', 
            color=OKABE_ITO['green'], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_ylabel('成本（元）', fontweight='bold')
ax.set_title('图3: 成本结构对比', fontweight='bold', pad=15)
ax.set_xticks(x_cat)
ax.set_xticklabels(cost_categories)
ax.legend(frameon=True, fancybox=False, edgecolor='black')

for i, total in enumerate([initial_total_cost, final_total_cost]):
    ax.text(i, total + 5, f'总计: {total:.1f}元', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color='#333333')

sns.despine()
plt.tight_layout()
plt.savefig('问题2_图3_成本结构对比.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图3_成本结构对比.png")
plt.close()

# ========================================================================
# 图4: 调度网络示意图（地理坐标+弧线连接）
# ========================================================================
print("[生成] 图4: 调度网络示意图...")

fig, ax = plt.subplots(figsize=(14, 10))

if len(schedule_df) > 0:
    station_coords = {}
    
    type_colors = {
        'residential': OKABE_ITO['sky_blue'],
        'business': OKABE_ITO['vermillion'],
        'education': OKABE_ITO['green'],
        'transport': OKABE_ITO['purple']
    }
    
    type_labels = {
        'residential': '居民区',
        'business': '商务区',
        'education': '教育区',
        'transport': '交通枢纽'
    }
    
    for idx, row_info in station_info.iterrows():
        sid = row_info['station_id']
        lon = row_info['经度']
        lat = row_info['纬度']
        stype = row_info['类型']
        sname = row_info['站点名称']
        
        station_coords[sid] = {'lon': lon, 'lat': lat, 'type': stype, 'name': sname}
        
        color = type_colors.get(stype, '#999999')
        ax.scatter(lon, lat, s=800, c=color, alpha=0.75, 
                  edgecolors='white', linewidth=2, zorder=5)
        ax.annotate(sname, (lon, lat), xytext=(0, 18), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=6)
    
    for _, sched_row in schedule_df.iterrows():
        start_id = sched_row['起点站点']
        end_id = sched_row['终点站点']
        num_bikes = int(sched_row['调度量'])
        truck_id = sched_row['卡车编号']
        
        start_lon = station_coords[start_id]['lon']
        start_lat = station_coords[start_id]['lat']
        end_lon = station_coords[end_id]['lon']
        end_lat = station_coords[end_id]['lat']
        
        color = truck_colors.get(truck_id, '#666666')
        
        rad = 0.25 if num_bikes >= 4 else 0.2
        
        ax.annotate('', xy=(end_lon, end_lat), xytext=(start_lon, start_lat),
                   arrowprops=dict(arrowstyle='->', lw=1.5+num_bikes*0.12, 
                                  color=color, alpha=0.8,
                                  connectionstyle=f'arc3,rad={rad}'),
                   zorder=3)
        
        t = 0.25
        
        point_on_line_x = (1-t)**2 * start_lon + 2*(1-t)*t * (start_lon + end_lon)/2 + t**2 * end_lon
        point_on_line_y = (1-t)**2 * start_lat + 2*(1-t)*t * (start_lat + end_lat)/2 + t**2 * end_lat
        
        dx = end_lon - start_lon
        dy = end_lat - start_lat
        length = np.sqrt(dx**2 + dy**2) if (dx != 0 or dy != 0) else 1
        
        offset_dist = 0.001 + num_bikes * 0.0002
        perp_x = -dy / length * offset_dist * np.sign(rad)
        perp_y = dx / length * offset_dist * np.sign(rad)
        
        label_x = point_on_line_x + perp_x
        label_y = point_on_line_y + perp_y
        
        fontsize = 7 if num_bikes <= 2 else 8
        ax.annotate(f'{truck_id}\n{num_bikes}辆',
                   xy=(point_on_line_x, point_on_line_y),
                   xytext=(label_x, label_y),
                   ha='center', va='center', fontsize=fontsize, fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.12', 
                             facecolor='white', edgecolor=color, 
                             alpha=0.95, linewidth=0.9),
                   arrowprops=dict(arrowstyle='-', color=color, lw=0.5,
                                   connectionstyle='arc3,rad=0'),
                   zorder=4)
    
    all_lons = [c['lon'] for c in station_coords.values()]
    all_lats = [c['lat'] for c in station_coords.values()]
    lon_margin = (max(all_lons) - min(all_lons)) * 0.1
    lat_margin = (max(all_lats) - min(all_lats)) * 0.15
    
    ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
    
    ax.set_xlabel('经度', fontweight='bold')
    ax.set_ylabel('纬度', fontweight='bold')
    ax.set_title('图4: 调度网络示意图\n（弧线表示调度路径，粗细表示调度量）', 
                fontweight='bold', pad=15)
    
    legend_elements = []
    
    seen_types = set()
    for stype, scolor in type_colors.items():
        if stype not in seen_types and stype in [c['type'] for c in station_coords.values()]:
            legend_elements.append(Patch(facecolor=scolor, edgecolor='white', 
                                        label=type_labels.get(stype, stype)))
            seen_types.add(stype)
    
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=0))
    
    for tid, tcolor in truck_colors.items():
        legend_elements.append(Line2D([0], [0], color=tcolor, lw=2.5, 
                                     marker='>', markersize=8, label=f'{tid} 卡车'))
    
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
             fancybox=False, edgecolor='black', fontsize=9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='datalim')
    
else:
    ax.text(0.5, 0.5, '无调度操作', ha='center', va='center', fontsize=16,
           transform=ax.transAxes)
    ax.set_title('图4: 调度网络示意图', fontweight='bold', pad=15)
    ax.axis('off')

sns.despine()
plt.tight_layout()
plt.savefig('问题2_图4_调度网络示意图.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图4_调度网络示意图.png")
plt.close()

# ========================================================================
# 图5: 各站点偏差改善
# ========================================================================
print("[生成] 图5: 各站点偏差改善...")

fig, ax = plt.subplots(figsize=(12, 6))
initial_dev = [abs(row['初始库存'] - row['期望库存']) for _, row in inventory_df.iterrows()]
final_dev = [abs(row['调度后库存'] - row['期望库存']) for _, row in inventory_df.iterrows()]

x_pos = np.arange(len(inventory_df))
width = 0.35

bars1 = ax.bar(x_pos - width/2, initial_dev, width, label='调度前偏差', 
               color=OKABE_ITO['vermillion'], alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, final_dev, width, label='调度后偏差', 
               color=OKABE_ITO['green'], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('站点', fontweight='bold')
ax.set_ylabel('偏差（辆）', fontweight='bold')
ax.set_title('图5: 各站点偏差改善', fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(inventory_df['站点'], rotation=45, ha='right')
ax.legend(frameon=True, fancybox=False, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=7)

sns.despine()
plt.tight_layout()
plt.savefig('问题2_图5_各站点偏差改善.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图5_各站点偏差改善.png")
plt.close()

# ========================================================================
# 图6: 卡车工作量分布
# ========================================================================
print("[生成] 图6: 卡车工作量分布...")

fig, ax = plt.subplots(figsize=(8, 6))

if len(schedule_df) > 0:
    truck_stats = schedule_df.groupby('卡车编号').agg({
        '调度量': 'sum',
        '距离(km)': 'sum',
        '时间(分钟)': 'sum',
        '运输成本(元)': 'sum'
    }).reset_index()

    x_pos = np.arange(len(truck_stats))
    colors = [truck_colors.get(t, '#999999') for t in truck_stats['卡车编号']]
    
    bars = ax.bar(x_pos, truck_stats['时间(分钟)'], color=colors, 
                 alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=T_max, color=OKABE_ITO['vermillion'], linestyle='--', 
              linewidth=2, label=f'时间上限 ({T_max}分钟)')
    
    for bar, (_, row) in zip(bars, truck_stats.iterrows()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
               f'{height:.1f}分\n({row["调度量"]}辆)', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('卡车编号', fontweight='bold')
    ax.set_ylabel('工作时间（分钟）', fontweight='bold')
    ax.set_title('图6: 卡车工作量分布', fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(truck_stats['卡车编号'])
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.set_ylim(0, T_max * 1.2)
else:
    ax.text(0.5, 0.5, '无调度操作', ha='center', va='center', fontsize=14,
           transform=ax.transAxes)
    ax.set_title('图6: 卡车工作量分布', fontweight='bold', pad=15)
    ax.axis('off')

sns.despine()
plt.tight_layout()
plt.savefig('问题2_图6_卡车工作量分布.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图6_卡车工作量分布.png")
plt.close()

# ========================================================================
# 图7: 综合性能评估（雷达图）
# ========================================================================
print("[生成] 图7: 综合性能评估...")

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

metrics = [
    satisfaction_rate_after,
    100 - (float(inventory_df['调度后缺货'].sum()) / max(float(inventory_df['调度前缺货'].sum()), 1) * 100),
    100 - (float(inventory_df['调度后积压'].sum()) / max(float(inventory_df['调度前积压'].sum()), 1) * 100),
    100 - (transport_cost_total / max(initial_total_cost, 1) * 100),
    100 - (float(schedule_df['时间(分钟)'].sum()) / T_max * 100) if len(schedule_df) > 0 else 100
]

labels = ['需求满足率', '缺货改善', '积压改善', '成本效率', '时间效率']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
metrics_plot = metrics + metrics[:1]
angles_plot = angles + angles[:1]

ax.plot(angles_plot, metrics_plot, 'o-', linewidth=2.5, color=OKABE_ITO['blue'], 
       markerfacecolor=OKABE_ITO['blue'], markersize=8, markeredgecolor='white', 
       markeredgewidth=2)
ax.fill(angles_plot, metrics_plot, alpha=0.25, color=OKABE_ITO['blue'])

ax.set_xticks(angles_plot[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
ax.set_title('图7: 综合性能评估', fontsize=13, fontweight='bold', pad=25)
ax.grid(True, linestyle='-', alpha=0.3)

for angle, metric, label in zip(angles, metrics, labels):
    ax.annotate(f'{metric:.1f}%', xy=(angle, metric), xytext=(angle, metric + 8),
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               color=OKABE_ITO['vermillion'])

plt.tight_layout()
plt.savefig('问题2_图7_综合性能评估.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_图7_综合性能评估.png")
plt.close()

print("\n" + "="*80)
print("所有可视化图表生成完成！")
print("="*80)
print("\n生成的图表:")
print("  - 问题2_图1_站点库存对比.png")
print("  - 问题2_图2_缺货积压对比.png")
print("  - 问题2_图3_成本结构对比.png")
print("  - 问题2_图4_调度网络示意图.png")
print("  - 问题2_图5_各站点偏差改善.png")
print("  - 问题2_图6_卡车工作量分布.png")
print("  - 问题2_图7_综合性能评估.png")