"""
问题二：多日期验证 & MILP vs 启发式对比
运行6月1-14日全部数据，输出对比结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OKABE_ITO = {
    'orange': '#E69F00', 'sky_blue': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00',
    'purple': '#CC79A7'
}

print("=" * 80)
print("问题二：多日期验证 + MILP vs 启发式算法对比")
print("=" * 80)

# ============================================================================
# 数据加载
# ============================================================================
station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8')
expected_inventory = pd.read_csv('站点期望库存.csv', encoding='utf-8')
initial_inventory = pd.read_csv('每日初始库存.csv', encoding='utf-8')
distance_matrix = pd.read_csv('站点距离.csv', index_col=0, encoding='utf-8')
weather_data = pd.read_csv('天气数据.csv', encoding='utf-8')

initial_inventory['日期'] = pd.to_datetime(initial_inventory['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])

stations = station_info['station_id'].tolist()
N = len(stations)

delta = dict(zip(expected_inventory['站点ID'], expected_inventory['允许偏差']))
C = dict(zip(station_info['station_id'], station_info['容量']))

dist = {}
for i in stations:
    dist[i] = {}
    for j in stations:
        dist[i][j] = distance_matrix.loc[i, j]

M = 3; Q = 15; v = 20
t_load = 1; t_unload = 1; T_max = 300
c_transport = 2.0; c_shortage = 10.0; c_excess = 5.0
trucks = [f'T{i+1}' for i in range(M)]

def calc_time(i, j, num_bikes):
    if i == j: return 0
    return (dist[i][j] / v) * 60 + num_bikes * (t_load + t_unload)

all_dates = sorted(initial_inventory['日期'].unique())
print(f"\n共 {len(all_dates)} 天数据: {all_dates[0].strftime('%m/%d')} ~ {all_dates[-1].strftime('%m/%d')}")


# ============================================================================
# MILP 求解器（单天）
# ============================================================================
def solve_milp_one_day(date):
    day_inventory = initial_inventory[initial_inventory['日期'] == date]
    I0 = {}
    for _, row in day_inventory.iterrows():
        for s in stations:
            if s in row.index:
                I0[s] = row[s]

    weather_info = weather_data[weather_data['日期'] == date].iloc[0]
    is_workday = 1 - weather_info['是否节假日']

    I_star = {}
    for _, row in expected_inventory.iterrows():
        station = row['站点ID']
        I_star[station] = row['工作日期望库存'] if is_workday else row['周末期望库存']

    prob = LpProblem("Bike_MILP", LpMinimize)

    x = {}; y = {}
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j:
                    x[k,i,j] = LpVariable(f"x_{k}_{i}_{j}", lowBound=0, upBound=Q, cat='Integer')
                    y[k,i,j] = LpVariable(f"y_{k}_{i}_{j}", cat='Binary')

    I = {i: LpVariable(f"I_{i}", lowBound=0, upBound=C[i], cat='Integer') for i in stations}
    shortage_var = {i: LpVariable(f"sh_{i}", lowBound=0) for i in stations}
    excess_var = {i: LpVariable(f"ex_{i}", lowBound=0) for i in stations}

    prob += lpSum([x[k,i,j] * dist[i][j] * c_transport
                   for k in trucks for i in stations for j in stations if i != j]) \
          + lpSum([shortage_var[i] * c_shortage for i in stations]) \
          + lpSum([excess_var[i] * c_excess for i in stations])

    for i in stations:
        prob += I[i] == I0[i] + lpSum([x[k,j,i] for k in trucks for j in stations if j != i]) \
                - lpSum([x[k,i,j] for k in trucks for j in stations if j != i])

    for i in stations:
        prob += shortage_var[i] >= (I_star[i] - delta[i]) - I[i]
        prob += excess_var[i] >= I[i] - (I_star[i] + delta[i])

    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j:
                    prob += x[k,i,j] <= Q * y[k,i,j]
                    prob += x[k,i,j] >= y[k,i,j]

    for i in stations:
        prob += lpSum([x[k,i,j] for k in trucks for j in stations if j != i]) <= I0[i]

    for k in trucks:
        prob += lpSum([y[k,i,j] * calc_time(i, j, Q/2)
                       for i in stations for j in stations if i != j]) <= T_max

    solver = PULP_CBC_CMD(msg=0, timeLimit=120)
    prob.solve(solver)

    if prob.status != 1:
        return None, None

    schedule = []
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j and value(y[k,i,j]) == 1:
                    num = int(value(x[k,i,j]))
                    schedule.append({
                        '卡车': k, '起点': i, '终点': j, '调度量': num,
                        '距离': dist[i][j], '时间': calc_time(i, j, num),
                        '运输成本': num * dist[i][j] * c_transport
                    })

    df = pd.DataFrame(schedule)
    transport_cost = df['运输成本'].sum() if len(df) > 0 else 0

    total_short = 0; total_exc = 0; n_satisfied = 0
    for i in stations:
        final = int(value(I[i]))
        dev = final - I_star[i]
        total_short += max(0, -dev - delta[i])
        total_exc += max(0, dev - delta[i])
        if abs(dev) <= delta[i]:
            n_satisfied += 1

    return {
        'date': date, 'workday': is_workday,
        'schedule_count': len(df), 'total_bikes': df['调度量'].sum() if len(df) > 0 else 0,
        'total_distance': df['距离'].sum() if len(df) > 0 else 0,
        'total_time': df['时间'].sum() if len(df) > 0 else 0,
        'transport_cost': transport_cost,
        'shortage_after': total_short, 'excess_after': total_exc,
        'shortage_cost': total_short * c_shortage, 'excess_cost': total_exc * c_excess,
        'total_cost': transport_cost + total_short * c_shortage + total_exc * c_excess,
        'satisfied_rate': n_satisfied / N * 100,
        'schedule_df': df
    }, I_star


# ============================================================================
# 启发式算法（单天）
# ============================================================================
def solve_heuristic_one_day(date, I_star):
    day_inventory = initial_inventory[initial_inventory['日期'] == date]
    I0 = {}
    for _, row in day_inventory.iterrows():
        for s in stations:
            if s in row.index:
                I0[s] = row[s]

    supply_demand = {}
    for s in stations:
        dev = I0[s] - I_star[s]
        if dev < -delta[s]:
            supply_demand[s] = dev + delta[s]
        elif dev > delta[s]:
            supply_demand[s] = dev - delta[s]
        else:
            supply_demand[s] = 0

    supply_stations = [(s, supply_demand[s]) for s in stations if supply_demand[s] > 0]
    demand_stations = [(s, -supply_demand[s]) for s in stations if supply_demand[s] < 0]

    supply_stations.sort(key=lambda x: x[1], reverse=True)
    demand_stations.sort(key=lambda x: x[1], reverse=True)

    all_pairs = []
    for ss, _ in supply_stations:
        for ds, _ in demand_stations:
            all_pairs.append((ss, ds, dist[ss][ds]))
    all_pairs.sort(key=lambda x: x[2])

    sup_rem = {s: a for s, a in supply_stations}
    dem_rem = {s: a for s, a in demand_stations}

    tasks = []
    for ss, ds, d in all_pairs:
        if sup_rem.get(ss, 0) > 0 and dem_rem.get(ds, 0) > 0:
            amt = min(sup_rem[ss], dem_rem[ds], Q)
            if amt >= 1:
                tasks.append({'from': ss, 'to': ds, 'amount': int(amt), 'distance': d,
                              'time': calc_time(ss, ds, int(amt))})
                sup_rem[ss] -= amt
                dem_rem[ds] -= amt

    truck_time = {t: 0 for t in trucks}
    truck_schedule = {t: [] for t in trucks}
    schedule = []

    for task in tasks:
        avail = [(t, truck_time[t]) for t in trucks if truck_time[t] + task['time'] <= T_max]
        if avail:
            chosen = min(avail, key=lambda x: x[1])[0]
            truck_schedule[chosen].append(task)
            truck_time[chosen] += task['time']
            schedule.append({
                '卡车': chosen, '起点': task['from'], '终点': task['to'],
                '调度量': task['amount'], '距离': task['distance'],
                '时间': task['time'],
                '运输成本': task['amount'] * task['distance'] * c_transport
            })

    df = pd.DataFrame(schedule)
    transport_cost = df['运输成本'].sum() if len(df) > 0 else 0

    I_final = I0.copy()
    for _, row in df.iterrows():
        I_final[row['起点']] -= row['调度量']
        I_final[row['终点']] += row['调度量']

    total_short = 0; total_exc = 0; n_satisfied = 0
    for s in stations:
        dev = I_final[s] - I_star[s]
        total_short += max(0, -dev - delta[s])
        total_exc += max(0, dev - delta[s])
        if abs(dev) <= delta[s]:
            n_satisfied += 1

    return {
        'date': date,
        'schedule_count': len(df), 'total_bikes': df['调度量'].sum() if len(df) > 0 else 0,
        'total_distance': df['距离'].sum() if len(df) > 0 else 0,
        'total_time': df['时间'].sum() if len(df) > 0 else 0,
        'transport_cost': transport_cost,
        'shortage_after': total_short, 'excess_after': total_exc,
        'shortage_cost': total_short * c_shortage, 'excess_cost': total_exc * c_excess,
        'total_cost': transport_cost + total_short * c_shortage + total_exc * c_excess,
        'satisfied_rate': n_satisfied / N * 100
    }


# ============================================================================
# 逐日运行
# ============================================================================
print("\n" + "=" * 80)
print("逐日运行中...")
print("-" * 80)

milp_results = []
heuristic_results = []

for idx, date in enumerate(all_dates):
    date_str = date.strftime('%m/%d')
    print(f"\n[{idx+1}/14] {date_str}...", end=' ')

    milp_r, I_star = solve_milp_one_day(date)
    if milp_r:
        milp_results.append(milp_r)
        heu_r = solve_heuristic_one_day(date, I_star)
        heuristic_results.append(heu_r)
        print(f"MILP:{milp_r['total_cost']:.1f}元 启发式:{heu_r['total_cost']:.1f}元 差距:{heu_r['total_cost']-milp_r['total_cost']:.1f}元")
    else:
        print("MILP未找到最优解，跳过")

milp_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'schedule_df'} for r in milp_results])
heu_df = pd.DataFrame(heuristic_results)

# ============================================================================
# 汇总报告
# ============================================================================
print("\n" + "=" * 80)
print("多日期验证汇总")
print("=" * 80)

milp_df['日期标签'] = [d.strftime('%m/%d') for d in milp_df['date']]
milp_df['是否工作日'] = ['工作日' if w else '周末' for w in milp_df['workday']]

heu_df['日期标签'] = [d.strftime('%m/%d') for d in heu_df['date']]

print(f"\nMILP模型 ({len(milp_df)}天):")
print(f"  日均调度次数: {milp_df['schedule_count'].mean():.1f} 次")
print(f"  日均调度量: {milp_df['total_bikes'].mean():.1f} 辆")
print(f"  日均总成本: {milp_df['total_cost'].mean():.1f} 元")
print(f"  日均运输成本: {milp_df['transport_cost'].mean():.1f} 元")
print(f"  平均需求满足率: {milp_df['satisfied_rate'].mean():.1f}%")
print(f"  成本标准差: {milp_df['total_cost'].std():.1f} 元")

print(f"\n启发式算法 ({len(heu_df)}天):")
print(f"  日均调度次数: {heu_df['schedule_count'].mean():.1f} 次")
print(f"  日均调度量: {heu_df['total_bikes'].mean():.1f} 辆")
print(f"  日均总成本: {heu_df['total_cost'].mean():.1f} 元")
print(f"  日均运输成本: {heu_df['transport_cost'].mean():.1f} 元")
print(f"  平均需求满足率: {heu_df['satisfied_rate'].mean():.1f}%")

print(f"\nMILP vs 启发式对比:")
avg_milp = milp_df['total_cost'].mean()
avg_heu = heu_df['total_cost'].mean()
print(f"  MILP平均成本: {avg_milp:.1f} 元")
print(f"  启发式平均成本: {avg_heu:.1f} 元")
print(f"  MILP优势: {avg_heu - avg_milp:.1f} 元/天 ({(avg_heu - avg_milp)/avg_heu*100:.1f}%更低)")

print(f"\n工作日 vs 周末差异:")
workday_milp = milp_df[milp_df['workday'] == 1]
weekend_milp = milp_df[milp_df['workday'] == 0]
print(f"  工作日MILP平均成本: {workday_milp['total_cost'].mean():.1f} 元")
print(f"  周末MILP平均成本: {weekend_milp['total_cost'].mean():.1f} 元")

# ============================================================================
# 保存数据
# ============================================================================
comparison_df = milp_df[['日期标签', '是否工作日', 'schedule_count', 'total_bikes',
    'total_distance', 'total_time', 'transport_cost', 'total_cost', 'satisfied_rate']].copy()
comparison_df.columns = ['日期', '日期类型', 'MILP调度次数', 'MILP调度量(辆)',
    'MILP总距离(km)', 'MILP总时间(分)', 'MILP运输成本(元)', 'MILP总成本(元)', 'MILP满足率(%)']

comparison_df2 = heu_df[['schedule_count', 'total_bikes', 'total_distance', 'total_time',
    'transport_cost', 'total_cost', 'satisfied_rate']].copy()
comparison_df2.columns = ['启发式调度次数', '启发式调度量(辆)', '启发式总距离(km)',
    '启发式总时间(分)', '启发式运输成本(元)', '启发式总成本(元)', '启发式满足率(%)']

full_comparison = pd.concat([comparison_df, comparison_df2], axis=1)
full_comparison.to_csv('问题2_多日期验证结果.csv', index=False, encoding='utf-8-sig')
print(f"\n[OK] 多日期验证结果已保存: 问题2_多日期验证结果.csv")

# ============================================================================
# 生成对比图表
# ============================================================================
print("\n生成对比图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：每日总成本对比（折线图）
ax = axes[0, 0]
x = range(len(milp_df))
ax.plot(x, milp_df['total_cost'], 'o-', color=OKABE_ITO['blue'], linewidth=2,
        markersize=6, label=f'MILP (均价{avg_milp:.0f}元)')
ax.plot(x, heu_df['total_cost'], 's--', color=OKABE_ITO['orange'], linewidth=2,
        markersize=6, label=f'启发式 (均价{avg_heu:.0f}元)')
ax.set_xticks(x)
ax.set_xticklabels(milp_df['日期标签'], rotation=45, fontsize=8)
ax.set_ylabel('总成本（元）', fontweight='bold')
ax.set_title('每日总成本对比', fontweight='bold')
ax.legend(frameon=True, edgecolor='black')
ax.grid(True, alpha=0.3)

# 图2：每日调度量对比
ax = axes[0, 1]
x_pos = np.arange(len(milp_df))
w = 0.35
ax.bar(x_pos - w/2, milp_df['total_bikes'], w, label='MILP',
       color=OKABE_ITO['blue'], alpha=0.85, edgecolor='black', linewidth=0.3)
ax.bar(x_pos + w/2, heu_df['total_bikes'], w, label='启发式',
       color=OKABE_ITO['orange'], alpha=0.85, edgecolor='black', linewidth=0.3)
ax.set_xticks(x_pos)
ax.set_xticklabels(milp_df['日期标签'], rotation=45, fontsize=8)
ax.set_ylabel('调度量（辆）', fontweight='bold')
ax.set_title('每日调度量对比', fontweight='bold')
ax.legend(frameon=True, edgecolor='black')
ax.grid(True, alpha=0.3, axis='y')

# 图3：满足率对比
ax = axes[1, 0]
ax.plot(x, milp_df['satisfied_rate'], 'o-', color=OKABE_ITO['green'], linewidth=2,
        markersize=6, label=f'MILP (平均{milp_df["satisfied_rate"].mean():.1f}%)')
ax.plot(x, heu_df['satisfied_rate'], 's--', color=OKABE_ITO['vermillion'], linewidth=2,
        markersize=6, label=f'启发式 (平均{heu_df["satisfied_rate"].mean():.1f}%)')
ax.set_xticks(x)
ax.set_xticklabels(milp_df['日期标签'], rotation=45, fontsize=8)
ax.set_ylabel('需求满足率（%）', fontweight='bold')
ax.set_title('每日需求满足率对比', fontweight='bold')
ax.legend(frameon=True, edgecolor='black')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

# 图4：成本结构堆叠
ax = axes[1, 1]
cost_comparison = pd.DataFrame({
    '指标': ['日均运输成本', '日均缺货成本', '日均积压成本'],
    'MILP': [milp_df['transport_cost'].mean(), milp_df['shortage_cost'].mean(), milp_df['excess_cost'].mean()],
    '启发式': [heu_df['transport_cost'].mean(), heu_df['shortage_cost'].mean(), heu_df['excess_cost'].mean()]
})
x_labels = cost_comparison['指标']
w2 = 0.35
x2 = np.arange(len(x_labels))
ax.bar(x2 - w2/2, cost_comparison['MILP'], w2, label='MILP',
       color=OKABE_ITO['blue'], alpha=0.85, edgecolor='black', linewidth=0.3)
ax.bar(x2 + w2/2, cost_comparison['启发式'], w2, label='启发式',
       color=OKABE_ITO['orange'], alpha=0.85, edgecolor='black', linewidth=0.3)
ax.set_xticks(x2)
ax.set_xticklabels(x_labels)
ax.set_ylabel('成本（元）', fontweight='bold')
ax.set_title('日均成本结构对比', fontweight='bold')
ax.legend(frameon=True, edgecolor='black')
for b2 in ax.patches:
    if b2.get_height() > 0:
        ax.text(b2.get_x() + b2.get_width()/2., b2.get_height() + 2,
                f'{b2.get_height():.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# 添加总标题
fig.suptitle('多日期验证结果：MILP vs 启发式算法对比（6月1-14日）',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('问题2_多日期验证对比.png', dpi=300, bbox_inches='tight')
print(f"[OK] 已保存: 问题2_多日期验证对比.png")
plt.close()

# ============================================================================
# 详细对比表（供论文使用）
# ============================================================================
print("\n" + "=" * 80)
print("详细逐日对比表")
print("=" * 80)

print(f"\n{'日期':<8} {'类型':<6} {'MILP成本':>10} {'启发式成本':>10} {'差距':>10} {'MILP满足率':>10} {'启发式满足率':>12}")
print("-" * 70)
for i in range(len(milp_df)):
    d = milp_df['日期标签'].iloc[i]
    t = milp_df['是否工作日'].iloc[i]
    mc = milp_df['total_cost'].iloc[i]
    hc = heu_df['total_cost'].iloc[i]
    gap = hc - mc
    ms = milp_df['satisfied_rate'].iloc[i]
    hs = heu_df['satisfied_rate'].iloc[i]
    print(f"{d:<8} {t:<6} {mc:>10.1f} {hc:>10.1f} {gap:>10.1f} {ms:>10.1f}% {hs:>11.1f}%")

print("-" * 70)
print(f"{'平均':<8} {'':<6} {avg_milp:>10.1f} {avg_heu:>10.1f} {avg_heu-avg_milp:>10.1f} "
      f"{milp_df['satisfied_rate'].mean():>10.1f}% {heu_df['satisfied_rate'].mean():>11.1f}%")

print(f"\n成本优势天数: {sum(1 for i in range(len(milp_df)) if milp_df['total_cost'].iloc[i] < heu_df['total_cost'].iloc[i])}/{len(milp_df)} 天")
print(f"MILP最优天数: {sum(1 for i in range(len(milp_df)) if milp_df['satisfied_rate'].iloc[i] >= heu_df['satisfied_rate'].iloc[i])}/{len(milp_df)} 天")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)