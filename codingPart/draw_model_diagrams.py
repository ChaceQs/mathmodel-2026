"""
数学建模论文 - 模型概念示意图（修复重叠版）
使用 ax.text + bbox 自动适配框大小，确保不重叠
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

C = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00',    'sky': '#56B4E9',   'purple': '#CC79A7',
    'dark': '#333333',   'gray': '#999999'
}


def box(label, color, fontsize=11, fontcolor='white', bold=True):
    """创建带边框的文本 - ax.text 会自动适配框大小"""
    return {
        's': label, 'fontsize': fontsize, 'fontweight': 'bold' if bold else 'normal',
        'color': fontcolor, 'ha': 'center', 'va': 'center',
        'bbox': dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor=color,
                     alpha=0.9, linewidth=1.5),
        'zorder': 5
    }


def arrow(ax, x1, y1, x2, y2, c='#555555', lw=2):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle='arc3,rad=0', mutation_scale=15),
                zorder=3)


def mini_text(ax, x, y, label):
    """小注释文本"""
    ax.text(x, y, label, ha='center', va='center', fontsize=8, color='#666666', zorder=4)


# ============================================================
# 图A: MILP 模型流程图 — 纵向流程，超大间距
# ============================================================
print("[1/3] MILP流程图...")

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# 顶部：输入数据——三个并列
y0 = 10.5
spacing = 3.0
for i, (label, color) in enumerate([
    ('站点信息\n距离矩阵', C['blue']),
    ('初始库存\n期望库存',   C['sky']),
    ('天气数据\n工作日/周末', C['purple'])
]):
    ax.text(2.0 + i * spacing, y0, **box(label, color, 10))

# 箭头
arrow(ax, 5.0, y0 - 0.6, 5.0, y0 - 1.3, C['dark'])

# 预处理
ax.text(5.0, y0 - 1.8, **box('数据预处理\n计算库存偏差 / 供给需求分类', C['sky'], 11))
arrow(ax, 5.0, y0 - 2.5, 5.0, y0 - 3.2, C['dark'])

# 模型构建
ax.text(5.0, y0 - 3.7, **box('MILP 模型构建', C['red'], 13))
mini_text(ax, 5.0, y0 - 4.3, '决策变量 x(k,i,j)  y(k,i,j)  目标函数 min Z  约束条件')
arrow(ax, 5.0, y0 - 4.7, 5.0, y0 - 5.4, C['dark'])

# 求解
ax.text(5.0, y0 - 5.9, **box('CBC 求解器求解\n状态: Optimal  时间: 0.01s  最优值: 242.20元', C['orange'], 10.5))
arrow(ax, 5.0, y0 - 6.6, 5.0, y0 - 7.3, C['dark'])

# 输出 — 三个并列
y_out = y0 - 7.8
for i, (label, color) in enumerate([
    ('调度方案表\n8次调度 · 24辆', C['green']),
    ('成本效益分析\n降低31.8%',       C['green']),
    ('多日期验证\n14天全通过',       C['green'])
]):
    ax.text(2.0 + i * spacing, y_out, **box(label, color, 10))

# 标题
ax.set_title('图A: MILP 调度优化模型流程图', fontsize=16, fontweight='bold',
             color=C['dark'], pad=25)

plt.tight_layout(pad=0.3)
plt.savefig('模型示意图_MILP流程图.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK]")


# ============================================================
# 图B: MPC 框架图 — 环形循环
# ============================================================
print("[2/3] MPC框架图...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 顶部左侧：历史数据
ax.text(1.2, 8.5, **box('历史数据\n(6/1 - 6/10)', C['blue'], 10))
mini_text(ax, 1.2, 7.8, '10天训练')

# 顶部右侧：模型训练
ax.text(4.5, 8.5, **box('XGBoost 模型训练\n26个特征', C['purple'], 10))
arrow(ax, 2.2, 8.5, 3.5, 8.5, C['dark'])

# 中间：四个步骤横向排列
cx, cy = 7.0, 5.5
step_data = [
    ('(1) 预测需求',    C['blue'],   'XGBoost预测\n各站点期望库存'),
    ('(2) MILP 优化',   C['red'],    '求解最优\n调度方案'),
    ('(3) 执行调度',    C['orange'], '卡车运输\n车辆调配'),
    ('(4) 观测库存',    C['green'],  '获取实际库存\n反馈更新'),
]
step_w = 2.8
for i, (title, col, desc) in enumerate(step_data):
    sx = 2.2 + i * 3.2
    ax.text(sx, cy, **box(title, col, 11))
    mini_text(ax, sx, cy - 0.7, desc)
    if i < 3:
        arrow(ax, sx + 1.0, cy, sx + 2.2, cy, C['dark'])

# 反馈回路（从步骤4下方回到步骤1下方）
ax.annotate('', xy=(2.0, cy - 1.5), xytext=(11.6, cy - 1.5),
            arrowprops=dict(arrowstyle='->', color=C['red'], lw=2.5,
                            connectionstyle='arc3,rad=0.4', mutation_scale=15),
            zorder=3)
ax.text(7.0, cy - 2.5, '更新历史数据 → 滚动下一天',
        ha='center', va='center', fontsize=10, fontweight='bold', color=C['red'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['red'], alpha=0.9))

# 时间线
tl_y = 2.0
for i, day in enumerate(['第11天', '第12天', '第13天', '第14天']):
    tx = 2.2 + i * 3.2
    ax.text(tx, tl_y, **box(day, C['sky'], 9, '#333333', False))
    if i < 3:
        arrow(ax, tx + 0.9, tl_y, tx + 2.3, tl_y, C['gray'], 1.5)

# 底部结果
ax.text(7.0, 0.8, **box('4天总成本: 1,422.76元    日均: 355.69元', C['green'], 11))

# 标题
ax.set_title('图B: MPC 滚动时域预测-调度一体化框架', fontsize=16, fontweight='bold',
             color=C['dark'], pad=20)

plt.tight_layout(pad=0.3)
plt.savefig('模型示意图_MPC框架图.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK]")


# ============================================================
# 图C: 整体技术路线图 — 三列并行
# ============================================================
print("[3/3] 技术路线图...")

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# 顶部标题
ax.text(8.0, 8.4, '共享单车调度优化与需求预测 — 技术路线图',
        ha='center', va='center', fontsize=16, fontweight='bold', color=C['dark'])

# 三个问题 —— 三列
cols = [2.5, 8.0, 13.5]
titles = [
    '问题一:\n数据特征分析与站点分类',
    '问题二:\n最优控制调度优化',
    '问题三:\n需求预测与集成应用'
]
tcolors = [C['blue'], C['red'], C['green']]

methods = [
    ('数据质量评估', '潮汐指数计算'),
    ('K-means 聚类',  '关键站点识别'),
    ('工作日/周末',   '天气影响分析'),

    ('MILP 精确建模', 'CBC 求解器'),
    ('启发式算法',    '多日期验证'),
    ('14天全量测试',  '敏感性分析'),

    ('XGBoost 预测',  '卡尔曼滤波'),
    ('MPC 滚动框架',  '预测-调度集成'),
    ('MAPE 1.04%',    'R² 0.998'),
]

results = [
    ('4类站点分类', '10个关键度排名'),
    ('成本降低31.8%', '满足率30%→80%'),
    ('MPC 4天1,422元', '鲁棒性提升'),
]

for pi in range(3):
    cx = cols[pi]

    # 标题
    ax.text(cx, 7.0, **box(titles[pi], tcolors[pi], 10))

    # 方法
    for j in range(3):
        m1, m2 = methods[pi * 3 + j]
        ax.text(cx - 1.1, 5.5 - j * 0.75, **box(m1, '#F0F0F0', 8.5, '#333333', True))
        ax.text(cx + 1.1, 5.5 - j * 0.75, **box(m2, '#F0F0F0', 8.5, '#333333', True))

    # 结果
    r1, r2 = results[pi]
    ax.text(cx - 1.1, 2.8, **box(r1, tcolors[pi], 9, 'white', True))
    ax.text(cx + 1.1, 2.8, **box(r2, tcolors[pi], 9, 'white', True))

# 三列之间的箭头
arrow(ax, 4.0, 6.2, 5.8, 6.2, C['gray'], 2.5)
arrow(ax, 9.5, 6.2, 11.3, 6.2, C['gray'], 2.5)
mini_text(ax, 4.9, 6.5, '特征输入')
mini_text(ax, 10.4, 6.5, '模型集成')

# 底部汇总
ax.text(8.0, 1.2, **box('综合评估: 成本降低31.8%  |  满足率80%+  |  MAPE 1.04%  |  14天全通过',
                          C['dark'], 11))

# 从三个结果到汇总的箭头
for col in cols:
    arrow(ax, col, 2.25, col, 1.75, C['gray'], 1.5)
    arrow(ax, col, 1.75, 8.0, 1.75, C['gray'], 1.0)

ax.plot([2.5, 13.5], [1.75, 1.75], '-', color=C['gray'], lw=1.2, alpha=0.4, zorder=1)

plt.tight_layout(pad=0.3)
plt.savefig('模型示意图_整体技术路线图.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("  [OK]")

print("\n" + "=" * 50)
print("三张示意图全部生成完成，无重叠！")
print("=" * 50)
