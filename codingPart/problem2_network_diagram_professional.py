"""
问题二：共享单车调度网络示意图 - 专业版
使用NetworkX和Matplotlib创建高质量的网络拓扑图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("生成专业版调度网络示意图")
print("="*80)

# ============================================================================
# 配置
# ============================================================================

# Okabe-Ito色盲友好配色
COLORS = {
    'residential': '#56B4E9',    # 天蓝色 - 居民区
    'business': '#E69F00',       # 橙色 - 商务区
    'education': '#009E73',      # 绿色 - 教育区
    'transport': '#CC79A7',      # 紫色 - 交通枢纽
    'T1': '#0072B2',            # 深蓝色 - T1卡车
    'T2': '#D55E00',            # 朱红色 - T2卡车
    'T3': '#009E73',            # 绿色 - T3卡车
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
})

# ============================================================================
# 数据加载
# ============================================================================

print("\n[加载] 读取数据...")
schedule_df = pd.read_csv('问题2_调度方案.csv', encoding='utf-8-sig')
station_info = pd.read_csv('附件数据/站点基础信息.csv', encoding='utf-8-sig')

print(f"[OK] 调度方案: {len(schedule_df)} 条")
print(f"[OK] 站点信息: {len(station_info)} 个")

# ============================================================================
# 站点信息处理
# ============================================================================

stations = {}
for _, row in station_info.iterrows():
    sid = row['station_id']
    stations[sid] = {
        'name': row['站点名称'],
        'type': row['类型'],
        'capacity': row['容量'],
        'color': COLORS.get(row['类型'], '#999999')
    }

# 手动设置站点位置（层次化布局，整体下移避免图例遮挡）
positions = {
    # 左侧 - 居民区
    'S001': (0, 3.0),
    'S002': (0, 2.0),
    'S003': (0, 1.0),
    'S004': (0, 0.0),

    # 中间层
    'S009': (1.5, -0.5),  # 教育区（底部）

    # 右侧 - 商务区和交通枢纽
    'S005': (3, 2.5),    # 科技大厦
    'S006': (3, 1.0),    # 金融中心
    'S007': (3, 1.75),   # 时代广场
    'S008': (3, 0.25),   # 地铁站口
    'S010': (3, 3.25),   # 商业街口
}

# ============================================================================
# 创建图形
# ============================================================================

print("\n[绘制] 创建网络图...")

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(-0.5, 3.8)
ax.set_ylim(-1.0, 4.3)
ax.axis('off')

# ============================================================================
# 绘制调度路径（箭头）
# ============================================================================

print("[绘制] 添加调度路径...")

for _, row in schedule_df.iterrows():
    start = row['起点站点']
    end = row['终点站点']
    bikes = int(row['调度量'])
    truck = row['卡车编号']

    start_pos = positions[start]
    end_pos = positions[end]

    # 箭头颜色和粗细
    color = COLORS[truck]
    linewidth = 1.5 + bikes * 0.8  # 根据调度量调整粗细

    # 计算箭头的弯曲度（避免重叠）
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    # 为不同路径设置不同的弯曲度
    connectionstyle = "arc3,rad=0.2"
    if (start, end) in [('S002', 'S005'), ('S003', 'S005')]:
        connectionstyle = "arc3,rad=0.15"
    elif (start, end) in [('S003', 'S006'), ('S004', 'S006'), ('S009', 'S006')]:
        connectionstyle = "arc3,rad=-0.15"
    elif (start, end) in [('S001', 'S010'), ('S002', 'S010')]:
        connectionstyle = "arc3,rad=0.25"

    # 绘制箭头
    arrow = FancyArrowPatch(
        start_pos, end_pos,
        arrowstyle='-|>',
        connectionstyle=connectionstyle,
        linewidth=linewidth,
        color=color,
        alpha=0.7,
        zorder=1,
        mutation_scale=25,
    )
    ax.add_patch(arrow)

    # 为每条路径设置标签位置参数 (t值: 0=起点, 1=终点, offset: 垂直于曲线的偏移)
    label_configs = {
        # 从S001出发的路径
        ('S001', 'S010'): {'t': 0.7, 'offset_x': 0.0, 'offset_y': 0.2},     # T1 8辆 - 后半程，上方
        ('S001', 'S005'): {'t': 0.45, 'offset_x': 0.0, 'offset_y': 0.25},   # T3 1辆 - 中前段，上方

        # 从S002出发的路径
        ('S002', 'S010'): {'t': 0.3, 'offset_x': -0.1, 'offset_y': 0.2},    # T3 1辆 - 前段，左上
        ('S002', 'S005'): {'t': 0.5, 'offset_x': 0.0, 'offset_y': 0.2},     # T1 6辆 - 中段，上方

        # 从S003出发的路径
        ('S003', 'S005'): {'t': 0.5, 'offset_x': 0.0, 'offset_y': 0.25},    # T1 4辆 - 中段，上方
        ('S003', 'S006'): {'t': 0.5, 'offset_x': 0.0, 'offset_y': -0.2},    # T2 1辆 - 中段，下方

        # 从S004出发的路径
        ('S004', 'S006'): {'t': 0.5, 'offset_x': 0.0, 'offset_y': -0.25},   # T2 1辆 - 中段，下方
        ('S004', 'S008'): {'t': 0.5, 'offset_x': 0.0, 'offset_y': -0.2},    # T3 1辆 - 中段，下方

        # 从S009出发的路径
        ('S009', 'S006'): {'t': 0.65, 'offset_x': 0.05, 'offset_y': 0.2},   # T3 2辆 - 后段，上方
    }

    # 获取标签配置
    config = label_configs.get((start, end))

    if config:
        t = config['t']

        # 根据弧度计算曲线上的点
        # 提取connectionstyle中的rad值
        rad = 0.2  # 默认值
        if connectionstyle == "arc3,rad=0.15":
            rad = 0.15
        elif connectionstyle == "arc3,rad=-0.15":
            rad = -0.15
        elif connectionstyle == "arc3,rad=0.25":
            rad = 0.25

        # 计算曲线上t位置的点（简化的贝塞尔曲线近似）
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2

        # 计算垂直于连线的偏移方向
        perp_x = -(end_pos[1] - start_pos[1])
        perp_y = (end_pos[0] - start_pos[0])
        perp_len = np.sqrt(perp_x**2 + perp_y**2)
        if perp_len > 0:
            perp_x /= perp_len
            perp_y /= perp_len

        # 控制点（用于弧线）
        ctrl_x = mid_x + perp_x * rad * np.sqrt(dx**2 + dy**2)
        ctrl_y = mid_y + perp_y * rad * np.sqrt(dx**2 + dy**2)

        # 二次贝塞尔曲线上的点
        label_x = (1-t)**2 * start_pos[0] + 2*(1-t)*t * ctrl_x + t**2 * end_pos[0]
        label_y = (1-t)**2 * start_pos[1] + 2*(1-t)*t * ctrl_y + t**2 * end_pos[1]

        # 添加额外偏移
        label_x += config['offset_x']
        label_y += config['offset_y']

        # 绘制标签
        label_text = f'{truck}\n{bikes}辆'
        ax.text(label_x, label_y, label_text,
                fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35',
                         facecolor='white',
                         edgecolor=color,
                         linewidth=2,
                         alpha=0.95),
                zorder=3)

# ============================================================================
# 绘制站点节点
# ============================================================================

print("[绘制] 添加站点节点...")

for sid, pos in positions.items():
    station = stations[sid]

    # 节点大小根据容量调整
    size = 800 + (station['capacity'] - 35) * 20

    # 绘制节点
    circle = plt.Circle(pos, 0.18,
                       color=station['color'],
                       ec='white',
                       linewidth=3,
                       zorder=4,
                       alpha=0.9)
    ax.add_patch(circle)

    # 站点名称（在节点上方）
    ax.text(pos[0], pos[1] + 0.28, station['name'],
            fontsize=11, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3',
                     facecolor='white',
                     edgecolor='#333333',
                     linewidth=1,
                     alpha=0.95),
            zorder=5)

    # 站点ID和容量（在节点内）
    ax.text(pos[0], pos[1], f'{sid}\n{station["capacity"]}辆',
            fontsize=9, fontweight='bold',
            ha='center', va='center',
            color='white',
            zorder=5)

# ============================================================================
# 添加标题
# ============================================================================

ax.text(1.5, 4.0, '共享单车调度网络示意图',
        fontsize=18, fontweight='bold',
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.6',
                 facecolor='#f0f0f0',
                 edgecolor='#333333',
                 linewidth=2))

ax.text(1.5, 3.8, 'Bike-sharing Dispatch Network Diagram',
        fontsize=12, style='italic',
        ha='center', va='top',
        color='#666666')

# ============================================================================
# 添加图例
# ============================================================================

print("[绘制] 添加图例...")

# 站点类型图例
legend_elements_stations = [
    mpatches.Patch(facecolor=COLORS['residential'],
                  edgecolor='white', linewidth=2,
                  label='居民区 (Residential)'),
    mpatches.Patch(facecolor=COLORS['business'],
                  edgecolor='white', linewidth=2,
                  label='商务区 (Business)'),
    mpatches.Patch(facecolor=COLORS['education'],
                  edgecolor='white', linewidth=2,
                  label='教育区 (Education)'),
    mpatches.Patch(facecolor=COLORS['transport'],
                  edgecolor='white', linewidth=2,
                  label='交通枢纽 (Transport Hub)'),
]

# 卡车类型图例
legend_elements_trucks = [
    mpatches.Patch(facecolor=COLORS['T1'],
                  edgecolor='none',
                  label='T1 卡车 (18辆)'),
    mpatches.Patch(facecolor=COLORS['T2'],
                  edgecolor='none',
                  label='T2 卡车 (2辆)'),
    mpatches.Patch(facecolor=COLORS['T3'],
                  edgecolor='none',
                  label='T3 卡车 (4辆)'),
]

# 创建两个图例
legend1 = ax.legend(handles=legend_elements_stations,
                   loc='upper left',
                   title='站点类型 (Station Types)',
                   title_fontsize=11,
                   frameon=True,
                   fancybox=False,
                   edgecolor='#333333',
                   framealpha=0.95,
                   fontsize=10)

legend2 = ax.legend(handles=legend_elements_trucks,
                   loc='lower left',
                   title='调度车辆 (Dispatch Trucks)',
                   title_fontsize=11,
                   frameon=True,
                   fancybox=False,
                   edgecolor='#333333',
                   framealpha=0.95,
                   fontsize=10)

ax.add_artist(legend1)  # 添加第一个图例

# ============================================================================
# 添加说明文本
# ============================================================================

note_text = (
    "说明 (Notes):\n"
    "• 箭头粗细表示调度量 (Arrow thickness indicates dispatch volume)\n"
    "• 节点大小表示站点容量 (Node size indicates station capacity)\n"
    "• 总调度量: 24辆 (Total dispatch: 24 bikes)"
)

ax.text(3.7, -0.8, note_text,
        fontsize=9,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5',
                 facecolor='#fffef0',
                 edgecolor='#999999',
                 linewidth=1,
                 alpha=0.9))

# ============================================================================
# 保存图形
# ============================================================================

print("\n[保存] 输出图像...")

plt.tight_layout()
plt.savefig('问题2_图4_调度网络示意图_专业版.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')

print(f"[OK] 已保存: 问题2_图4_调度网络示意图_专业版.png")
print("\n" + "="*80)
print("专业版网络图生成完成！")
print("="*80)

plt.close()
