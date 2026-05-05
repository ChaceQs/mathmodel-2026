"""论文加分图：灵敏度分析 + 鲁棒性检验"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')
from pulp import *
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

C = {'blue':'#0072B2','orange':'#E69F00','green':'#009E73','red':'#D55E00',
     'purple':'#CC79A7','sky':'#56B4E9','dark':'#333333','gray':'#999999'}

# ==================== 全局数据加载 ====================
station_info = pd.read_csv('附件数据/站点基础信息.csv', encoding='utf-8')
expected_inv = pd.read_csv('附件数据/站点期望库存.csv', encoding='utf-8')
initial_inv = pd.read_csv('附件数据/每日初始库存.csv', encoding='utf-8')
distance_matrix = pd.read_csv('附件数据/站点距离.csv', index_col=0, encoding='utf-8')
weather_data = pd.read_csv('附件数据/天气数据.csv', encoding='utf-8')

initial_inv['日期'] = pd.to_datetime(initial_inv['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])

last_date = initial_inv['日期'].max()
weather_info = weather_data[weather_data['日期'] == last_date].iloc[0]
IS_WORKDAY = 1 - weather_info['是否节假日']

stations = station_info['station_id'].tolist(); N = len(stations)
delta = dict(zip(expected_inv['站点ID'], expected_inv['允许偏差']))
capacity = dict(zip(station_info['station_id'], station_info['容量']))

T0 = {}
for _, row in initial_inv[initial_inv['日期'] == last_date].iterrows():
    for s in stations:
        if s in row.index: T0[s] = row[s]

STAR = {}
for _, row in expected_inv.iterrows():
    s = row['站点ID']
    STAR[s] = row['工作日期望库存'] if IS_WORKDAY else row['周末期望库存']

dist = {}
for i in stations:
    dist[i] = {}
    for j in stations: dist[i][j] = distance_matrix.loc[i, j]

def solve_milp(Q_val, c_trans, T_max, I0_dict, I_star_dict):
    """MILP求解器 — 可调参数版"""
    trucks = ['T1','T2','T3']; M = 3; v = 20
    t_load = 1; t_unload = 1; c_short = 10.0; c_exc = 5.0
    def calc_time(i, j, num):
        if i == j: return 0
        return (dist[i][j]/v)*60 + num*(t_load + t_unload)

    prob = LpProblem("SA", LpMinimize)
    x = {}; y = {}
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j:
                    x[k,i,j] = LpVariable(f"x_{k}_{i}_{j}", 0, Q_val, cat='Integer')
                    y[k,i,j] = LpVariable(f"y_{k}_{i}_{j}", cat='Binary')
    I = {i: LpVariable(f"I_{i}", 0, capacity[i], cat='Integer') for i in stations}
    sh = {i: LpVariable(f"sh_{i}", 0) for i in stations}
    ex = {i: LpVariable(f"ex_{i}", 0) for i in stations}

    prob += lpSum([x[k,i,j]*dist[i][j]*c_trans for k in trucks for i in stations for j in stations if i!=j]) \
          + lpSum([sh[i]*c_short for i in stations]) \
          + lpSum([ex[i]*c_exc for i in stations])

    for i in stations:
        prob += I[i] == I0_dict[i] + lpSum([x[k,j,i] for k in trucks for j in stations if j!=i]) \
                - lpSum([x[k,i,j] for k in trucks for j in stations if j!=i])
    for i in stations:
        prob += sh[i] >= (I_star_dict[i]-delta[i]) - I[i]
        prob += ex[i] >= I[i] - (I_star_dict[i]+delta[i])
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j:
                    prob += x[k,i,j] <= Q_val * y[k,i,j]
                    prob += x[k,i,j] >= y[k,i,j]
    for i in stations:
        prob += lpSum([x[k,i,j] for k in trucks for j in stations if j!=i]) <= I0_dict[i]
    for k in trucks:
        prob += lpSum([y[k,i,j]*calc_time(i,j,Q_val/2) for i in stations for j in stations if j!=i]) <= T_max

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    if prob.status == 1:
        transport = 0; short_c = 0; exc_c = 0
        for k in trucks:
            for i in stations:
                for j in stations:
                    if i != j:
                        nb = int(value(x[k,i,j]))
                        transport += nb * dist[i][j] * c_trans
        for i in stations:
            short_c += value(sh[i]) * c_short
            exc_c += value(ex[i]) * c_exc
        return transport + short_c + exc_c
    return None

print("=" * 60)
print("灵敏度分析 & 鲁棒性检验")
print("=" * 60)

# ============================================================
# 图A: 多参数灵敏度曲线（3个子图并排）
# ============================================================
print("\n[1/4] 多参数灵敏度曲线...")

base_Q = 15; base_c = 2.0; base_T = 300

# 参数1: 卡车容量 10 → 22
Q_range = list(range(10, 23, 2))
Q_costs = [solve_milp(q, base_c, base_T, T0, STAR) for q in Q_range]

# 参数2: 运输成本因子 0.3x → 2.0x
c_factors = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
c_costs = [solve_milp(base_Q, base_c*f, base_T, T0, STAR) for f in c_factors]

# 参数3: 时间窗口 120 → 480
T_range = [120, 180, 240, 300, 360, 420, 480]
T_costs = [solve_milp(base_Q, base_c, t, T0, STAR) for t in T_range]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(Q_range, Q_costs, 'o-', color=C['blue'], lw=2.5, ms=8, mfc='white', mew=2)
ax.axvline(x=15, color=C['red'], ls='--', lw=1.5, alpha=0.6, label=f'基准 Q={base_Q}')
ax.set_xlabel('卡车容量（辆）', fontweight='bold')
ax.set_ylabel('总成本（元）', fontweight='bold')
ax.set_title('卡车容量灵敏度', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(c_factors, c_costs, 's-', color=C['orange'], lw=2.5, ms=8, mfc='white', mew=2)
ax.axvline(x=1.0, color=C['red'], ls='--', lw=1.5, alpha=0.6, label='基准 1.0x')
ax.set_xlabel('运输成本系数', fontweight='bold')
ax.set_ylabel('总成本（元）', fontweight='bold')
ax.set_title('运输成本灵敏度', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(T_range, T_costs, 'D-', color=C['green'], lw=2.5, ms=8, mfc='white', mew=2)
ax.axvline(x=300, color=C['red'], ls='--', lw=1.5, alpha=0.6, label=f'基准 T={base_T}')
ax.set_xlabel('时间窗口（分钟）', fontweight='bold')
ax.set_ylabel('总成本（元）', fontweight='bold')
ax.set_title('时间窗口灵敏度', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle('图S1: 关键参数灵敏度分析', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('论文加分_图S1_灵敏度分析.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('  [OK] 论文加分_图S1_灵敏度分析.png')

# ============================================================
# 图B: 运输成本 × 卡车容量 二维热力图
# ============================================================
print("[2/4] 二维热力图...")

Q_grid = [10, 12, 14, 15, 16, 18, 20]
C_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
heatmap_data = np.zeros((len(Q_grid), len(C_grid)))
for i, q in enumerate(Q_grid):
    for j, cf in enumerate(C_grid):
        cost = solve_milp(q, base_c*cf, base_T, T0, STAR)
        heatmap_data[i, j] = cost if cost else np.nan
        print(f'    Q={q} c={cf}x => {cost:.1f}')

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower',
               extent=[C_grid[0]-0.25, C_grid[-1]+0.25, Q_grid[0]-1, Q_grid[-1]+1])
for i in range(len(Q_grid)):
    for j in range(len(C_grid)):
        ax.text(C_grid[j], Q_grid[i], f'{heatmap_data[i,j]:.0f}', ha='center', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('运输成本系数', fontweight='bold', fontsize=12)
ax.set_ylabel('卡车容量（辆）', fontweight='bold', fontsize=12)
ax.set_title('图S2: 运输成本 × 卡车容量 二维热力图\n（颜色越亮 = 总成本越高）', fontweight='bold', fontsize=13)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('总成本（元）', fontweight='bold')
ax.axhline(y=15, color='white', ls='--', lw=1.5, alpha=0.5)
ax.axvline(x=1.0, color='white', ls='--', lw=1.5, alpha=0.5)
plt.tight_layout()
plt.savefig('论文加分_图S2_二维热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('  [OK] 论文加分_图S2_二维热力图.png')

# ============================================================
# 图C: 鲁棒性箱线图 — 初始库存随机扰动
# ============================================================
print("[3/4] 鲁棒性箱线图...")

np.random.seed(42)
perturb_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
N_TRIALS = 20
robustness_data = {}

for level in perturb_levels:
    costs = []
    for trial in range(N_TRIALS):
        I0_perturbed = T0.copy()
        for s in stations:
            noise = np.random.normal(0, level * max(1, I0_perturbed[s]))
            I0_perturbed[s] = max(0, int(round(I0_perturbed[s] + noise)))
        cost = solve_milp(base_Q, base_c, base_T, I0_perturbed, STAR)
        if cost: costs.append(cost)
    robustness_data[f'{int(level*100)}%'] = costs
    print(f'  扰动{int(level*100):2d}%: mean={np.mean(costs):.1f} std={np.std(costs):.1f} min={min(costs):.1f} max={max(costs):.1f}')

fig, ax = plt.subplots(figsize=(10, 6))
positions = list(range(len(perturb_levels)))
bp = ax.boxplot([robustness_data[f'{int(l*100)}%'] for l in perturb_levels],
                positions=positions, widths=0.5, patch_artist=True,
                medianprops={'color':'black','lw':2},
                flierprops={'marker':'o','markersize':5,'markerfacecolor':C['red']})

colors_box = [C['blue'], C['sky'], C['green'], C['orange'], C['red']]
for patch, col in zip(bp['boxes'], colors_box):
    patch.set_facecolor(col); patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels([f'{int(l*100)}%' for l in perturb_levels])
ax.set_xlabel('初始库存扰动幅度', fontweight='bold', fontsize=12)
ax.set_ylabel('总成本（元）', fontweight='bold', fontsize=12)
ax.set_title('图S3: 初始库存扰动鲁棒性检验\n（20次随机模拟 / 扰动级别）', fontweight='bold', fontsize=13)
base_cost = robustness_data['0%'][0] if robustness_data['0%'] else 0
ax.axhline(y=base_cost, color=C['dark'], ls='--', lw=1.5, alpha=0.5, label=f'无扰动基准: {base_cost:.0f}元')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('论文加分_图S3_鲁棒性箱线图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('  [OK] 论文加分_图S3_鲁棒性箱线图.png')

# ============================================================
# 图D: 14天滚动鲁棒性 — 带扰动的成本分布
# ============================================================
print("[4/4] 14天滚动鲁棒性...")

all_dates = sorted(initial_inv['日期'].unique())
daily_costs_perturbed = {f'{int(l*100)}%': [] for l in [0.0, 0.05, 0.10, 0.15]}

for level in [0.0, 0.05, 0.10, 0.15]:
    label_key = f'{int(level*100)}%'
    for date in all_dates:
        day_inv = initial_inv[initial_inv['日期'] == date]
        I0_day = {}
        for _, row in day_inv.iterrows():
            for s in stations:
                if s in row.index: I0_day[s] = row[s]
        wi = weather_data[weather_data['日期'] == date].iloc[0]
        is_wd = 1 - wi['是否节假日']
        STAR_day = {}
        for _, row in expected_inv.iterrows():
            s = row['站点ID']
            STAR_day[s] = row['工作日期望库存'] if is_wd else row['周末期望库存']

        if level > 0:
            for s in stations:
                noise = np.random.normal(0, level * max(1, I0_day[s]))
                I0_day[s] = max(0, int(round(I0_day[s] + noise)))

        cost = solve_milp(base_Q, base_c, base_T, I0_day, STAR_day)
        if cost: daily_costs_perturbed[label_key].append(cost)
    vals = daily_costs_perturbed[label_key]
    print(f'  扰动{label_key}: mean={np.mean(vals):.1f} std={np.std(vals):.1f}')

fig, ax = plt.subplots(figsize=(12, 6))
bp2 = ax.boxplot([daily_costs_perturbed[f'{int(l*100)}%'] for l in [0.0, 0.05, 0.10, 0.15]],
                 positions=[0, 1, 2, 3], widths=0.5, patch_artist=True,
                 medianprops={'color':'black','lw':2},
                 flierprops={'marker':'o','markersize':5,'markerfacecolor':C['red']})
for patch, col in zip(bp2['boxes'], colors_box[:4]):
    patch.set_facecolor(col); patch.set_alpha(0.7)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['0%','5%','10%','15%'])
ax.set_xlabel('初始库存扰动幅度', fontweight='bold', fontsize=12)
ax.set_ylabel('总成本（元）', fontweight='bold', fontsize=12)
ax.set_title('图S4: 14天滚动鲁棒性检验\n（每天独立扰动 × 3次模拟）', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

stats_text = '无扰动: 均值{:.0f} std{:.0f}\n5%扰动: 均值{:.0f} std{:.0f}\n10%扰动: 均值{:.0f} std{:.0f}\n15%扰动: 均值{:.0f} std{:.0f}'.format(
    np.mean(daily_costs_perturbed['0%']), np.std(daily_costs_perturbed['0%']),
    np.mean(daily_costs_perturbed['5%']), np.std(daily_costs_perturbed['5%']),
    np.mean(daily_costs_perturbed['10%']), np.std(daily_costs_perturbed['10%']),
    np.mean(daily_costs_perturbed['15%']), np.std(daily_costs_perturbed['15%']))
ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['gray'], alpha=0.85, linewidth=0.5))
plt.tight_layout()
plt.savefig('论文加分_图S4_14天滚动鲁棒性.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('  [OK] 论文加分_图S4_14天滚动鲁棒性.png')

print("\n" + "=" * 60)
print("4张加分图全部生成完成")
print("=" * 60)
