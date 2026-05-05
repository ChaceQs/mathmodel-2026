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
from matplotlib.patches import Patch, FancyArrowPatch
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
# 图4: 调度网络示意图（圆形布局 + 精确标签定位 + 防重叠）
# ========================================================================
print("[生成] 图4: 调度网络示意图...")

fig, ax = plt.subplots(figsize=(13, 13))
ax.set_aspect('equal')
ax.axis('off')

if len(schedule_df) > 0:
    station_list = station_info['station_id'].tolist()
    n = len(station_list)

    type_order = {'residential': 0, 'business': 1, 'education': 2, 'transport': 3}
    station_info_copy = station_info.copy()
    station_info_copy['_order'] = station_info_copy['类型'].map(type_order)
    station_info_sorted = station_info_copy.sort_values(['_order', 'station_id'])
    stations_ordered = station_info_sorted['station_id'].tolist()

    R_dot = 4.5
    R_name = 5.5

    station_pos = {}
    station_data = {}
    for i, sid in enumerate(stations_ordered):
        angle = 2 * np.pi * i / n - np.pi / 2
        station_pos[sid] = {'x': R_dot * np.cos(angle), 'y': R_dot * np.sin(angle),
                            'angle': angle}
        row = station_info[station_info['station_id'] == sid].iloc[0]
        station_data[sid] = {'name': row['站点名称'], 'type': row['类型']}

    type_colors = {
        'residential': OKABE_ITO['sky_blue'],
        'business': OKABE_ITO['vermillion'],
        'education': OKABE_ITO['green'],
        'transport': OKABE_ITO['purple']
    }

    type_labels_map = {
        'residential': '居民区', 'business': '商务区',
        'education': '教育区', 'transport': '交通枢纽'
    }

    for sid in stations_ordered:
        p = station_pos[sid]
        stype = station_data[sid]['type']
        sname = station_data[sid]['name']
        color = type_colors.get(stype, '#999999')

        ax.scatter(p['x'], p['y'], s=900, c=color, alpha=0.85,
                   edgecolors='white', linewidth=2.5, zorder=10)

        short_id = sid.replace('S0', '').replace('S', '')
        ax.text(p['x'], p['y'], short_id, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=11)

        a = station_pos[sid]['angle']
        nx = R_name * np.cos(a)
        ny = R_name * np.sin(a)
        ax.text(nx, ny, sname, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                         edgecolor=color, alpha=0.9, linewidth=1.5),
                zorder=9)

    placed_labels = []

    for _, sched_row in schedule_df.iterrows():
        sid = sched_row['起点站点']
        eid = sched_row['终点站点']
        num_bikes = int(sched_row['调度量'])
        truck_id = sched_row['卡车编号']

        sx = station_pos[sid]['x']
        sy = station_pos[sid]['y']
        ex = station_pos[eid]['x']
        ey = station_pos[eid]['y']
        color = truck_colors.get(truck_id, '#666666')

        rad = 0.35 if num_bikes >= 4 else 0.22

        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle='->,head_length=0.35,head_width=0.25',
            connectionstyle=f'arc3,rad={rad}',
            lw=2.0 + num_bikes * 0.18, color=color, alpha=0.82, zorder=5
        )
        ax.add_patch(arrow)

        mx = (sx + ex) / 2.0
        my = (sy + ey) / 2.0
        dx = ex - sx
        dy = ey - sy
        chord_len = np.sqrt(dx**2 + dy**2)

        if chord_len > 0.01:
            nx_dir = -dy / chord_len
            ny_dir = dx / chord_len
            arc_offset = rad * chord_len
            arc_mx = mx + nx_dir * arc_offset
            arc_my = my + ny_dir * arc_offset
            label_dist = chord_len * 0.06 + 0.30
            lx = arc_mx + nx_dir * label_dist
            ly = arc_my + ny_dir * label_dist
        else:
            arc_mx, arc_my = mx, my
            lx, ly = mx, my + 0.6

        for px, py in placed_labels:
            if np.sqrt((lx - px)**2 + (ly - py)**2) < 1.3:
                lx += nx_dir * 0.7
                ly += ny_dir * 0.7
                break

        placed_labels.append((lx, ly))

        fontsize = 8 if num_bikes <= 2 else 9
        ax.text(lx, ly, f'{truck_id} {num_bikes}辆',
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=color, zorder=12,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                         edgecolor=color, alpha=0.95, linewidth=1.0))

        ax.plot([arc_mx, lx], [arc_my, ly], '-', color=color,
                lw=0.6, alpha=0.5, zorder=4)

    ax.set_title('图4: 调度网络示意图', fontsize=16, fontweight='bold', pad=15)

    legend_handles = []
    seen_types = set()
    for sid in stations_ordered:
        st = station_data[sid]['type']
        if st not in seen_types:
            legend_handles.append(Patch(facecolor=type_colors.get(st, '#999'),
                                       edgecolor='white', linewidth=1.5,
                                       label=type_labels_map.get(st, st)))
            seen_types.add(st)

    legend_handles.append(Line2D([0], [0], color='white', linewidth=0))
    for tid in ['T1', 'T2', 'T3']:
        legend_handles.append(Line2D([0], [0], color=truck_colors[tid], lw=3,
                                    label=tid))

    ax.legend(handles=legend_handles, loc='lower right',
             frameon=True, fancybox=False, edgecolor='#cccccc',
             fontsize=10, ncol=2)

    ax.set_xlim(-7.2, 7.2)
    ax.set_ylim(-7.2, 7.2)

else:
    ax.text(0.5, 0.5, '无调度操作', ha='center', va='center', fontsize=16,
           transform=ax.transAxes)
    ax.set_title('图4: 调度网络示意图', fontweight='bold', pad=15)

plt.tight_layout(pad=0.5)
plt.savefig('问题2_图4_调度网络示意图.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
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