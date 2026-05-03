"""
问题二：共享单车调度优化（完整实现）
合并版：数据加载、预处理、MILP模型、求解、结果分析、可视化

赛题要求：
- 3辆卡车，容量15辆
- 时间窗口：00:00-05:00（300分钟）
- 速度：20 km/h
- 装卸时间：1分钟/辆
- 成本：运输2元/km·辆，缺货10元/辆·次，积压5元/辆·次
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("问题二：共享单车调度优化（MILP模型）")
print("="*80)

# ============================================================================
# 第一部分：数据加载与预处理
# ============================================================================
print("\n[第一部分] 数据加载与预处理")
print("-"*80)

station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8')
station_features = pd.read_csv('站点特征分析结果.csv', encoding='utf-8')
expected_inventory = pd.read_csv('站点期望库存.csv', encoding='utf-8')
initial_inventory = pd.read_csv('每日初始库存.csv', encoding='utf-8')
distance_matrix = pd.read_csv('站点距离.csv', index_col=0, encoding='utf-8')
weather_data = pd.read_csv('天气数据.csv', encoding='utf-8')

print(f"[OK] 数据加载完成")
print(f"  - 站点数量: {len(station_info)}")
print(f"  - 数据天数: {len(initial_inventory)}")

initial_inventory['日期'] = pd.to_datetime(initial_inventory['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])

last_date = initial_inventory['日期'].max()
print(f"\n优化日期: {last_date.strftime('%Y-%m-%d')}")

weather_info = weather_data[weather_data['日期'] == last_date].iloc[0]
is_holiday = weather_info['是否节假日']
is_workday = 1 - is_holiday

print(f"  - 是否工作日: {'是' if is_workday else '否'}")
print(f"  - 天气: {weather_info['天气类型']}")

# ============================================================================
# 第二部分：准备优化模型参数
# ============================================================================
print("\n[第二部分] 准备优化模型参数")
print("-"*80)

stations = station_info['station_id'].tolist()
N = len(stations)

last_day_inventory = initial_inventory[initial_inventory['日期'] == last_date]
I0 = {}
for _, row in last_day_inventory.iterrows():
    for station in stations:
        if station in row.index:
            I0[station] = row[station]

I_star = {}
for _, row in expected_inventory.iterrows():
    station = row['站点ID']
    if is_workday:
        I_star[station] = row['工作日期望库存']
    else:
        I_star[station] = row['周末期望库存']

delta = dict(zip(expected_inventory['站点ID'], expected_inventory['允许偏差']))
C = dict(zip(station_info['station_id'], station_info['容量']))

weights_raw = dict(zip(station_features['station_id'], station_features['关键度评分']))
max_weight = max(weights_raw.values())
weights = {k: v/max_weight for k, v in weights_raw.items()}

dist = {}
for i in stations:
    dist[i] = {}
    for j in stations:
        dist[i][j] = distance_matrix.loc[i, j]

M = 3
trucks = [f'T{i+1}' for i in range(M)]
Q = 15

v = 20
t_load_per_bike = 1
t_unload_per_bike = 1
T_max = 300

def calc_time(i, j, num_bikes):
    if i == j:
        return 0
    travel_time = (dist[i][j] / v) * 60
    loading_time = num_bikes * t_load_per_bike
    unloading_time = num_bikes * t_unload_per_bike
    return travel_time + loading_time + unloading_time

c_transport = 2.0
c_shortage = 10.0
c_excess = 5.0

print(f"\n模型参数:")
print(f"  - 站点数: {N}")
print(f"  - 卡车数: {M}")
print(f"  - 卡车容量: {Q} 辆")
print(f"  - 时间窗口: {T_max} 分钟")
print(f"  - 平均速度: {v} km/h")
print(f"  - 装卸时间: {t_load_per_bike} 分钟/辆")
print(f"  - 运输成本: {c_transport} 元/km·辆")
print(f"  - 缺货成本: {c_shortage} 元/辆·次")
print(f"  - 积压成本: {c_excess} 元/辆·次")

print(f"\n初始库存状态:")
total_shortage = 0
total_excess = 0
for station in stations:
    deviation = I0[station] - I_star[station]
    if deviation < -delta[station]:
        shortage = abs(deviation) - delta[station]
        total_shortage += shortage
        print(f"  {station}: {I0[station]:2d}辆 (期望{I_star[station]:2d}辆, 缺货{shortage:.1f}辆)")
    elif deviation > delta[station]:
        excess = deviation - delta[station]
        total_excess += excess
        print(f"  {station}: {I0[station]:2d}辆 (期望{I_star[station]:2d}辆, 积压{excess:.1f}辆)")
    else:
        print(f"  {station}: {I0[station]:2d}辆 (期望{I_star[station]:2d}辆, 正常)")

print(f"\n调度前成本:")
print(f"  - 缺货成本: {total_shortage * c_shortage:.2f} 元")
print(f"  - 积压成本: {total_excess * c_excess:.2f} 元")
print(f"  - 总成本: {total_shortage * c_shortage + total_excess * c_excess:.2f} 元")

# ============================================================================
# 第三部分：构建MILP优化模型
# ============================================================================
print("\n[第三部分] 构建MILP优化模型")
print("-"*80)

print("开始构建MILP模型...")
prob = LpProblem("Bike_Scheduling_Optimization", LpMinimize)

x = {}
y = {}
for k in trucks:
    for i in stations:
        for j in stations:
            if i != j:
                x[k,i,j] = LpVariable(f"x_{k}_{i}_{j}", lowBound=0, upBound=Q, cat='Integer')
                y[k,i,j] = LpVariable(f"y_{k}_{i}_{j}", cat='Binary')

I = {i: LpVariable(f"I_{i}", lowBound=0, upBound=C[i], cat='Integer') for i in stations}
shortage_var = {i: LpVariable(f"shortage_{i}", lowBound=0) for i in stations}
excess_var = {i: LpVariable(f"excess_{i}", lowBound=0) for i in stations}

print(f"[OK] 决策变量创建完成")
print(f"  - 调度量变量 x: {len(x)}")
print(f"  - 二进制变量 y: {len(y)}")
print(f"  - 库存变量 I: {len(I)}")

print("\n构建目标函数...")

transport_cost = lpSum([x[k,i,j] * dist[i][j] * c_transport
                        for k in trucks for i in stations for j in stations if i != j])
shortage_cost = lpSum([shortage_var[i] * c_shortage for i in stations])
excess_cost = lpSum([excess_var[i] * c_excess for i in stations])

prob += transport_cost + shortage_cost + excess_cost, "Total_Cost"

print(f"[OK] 目标函数构建完成")

print("\n添加约束条件...")

for i in stations:
    prob += I[i] == I0[i] + lpSum([x[k,j,i] for k in trucks for j in stations if j != i]) \
                          - lpSum([x[k,i,j] for k in trucks for j in stations if j != i]), \
            f"Inventory_Balance_{i}"

for i in stations:
    prob += shortage_var[i] >= (I_star[i] - delta[i]) - I[i], f"Shortage_Def_{i}"
    prob += excess_var[i] >= I[i] - (I_star[i] + delta[i]), f"Excess_Def_{i}"

for k in trucks:
    for i in stations:
        for j in stations:
            if i != j:
                prob += x[k,i,j] <= Q * y[k,i,j], f"Capacity_{k}_{i}_{j}"

for k in trucks:
    for i in stations:
        for j in stations:
            if i != j:
                prob += x[k,i,j] >= y[k,i,j], f"Logic_{k}_{i}_{j}"

for i in stations:
    prob += lpSum([x[k,i,j] for k in trucks for j in stations if j != i]) <= I0[i], \
            f"Feasibility_{i}"

for k in trucks:
    avg_bikes = Q / 2
    prob += lpSum([y[k,i,j] * calc_time(i, j, avg_bikes)
                   for i in stations for j in stations if i != j]) <= T_max, \
            f"Time_Window_{k}"

print(f"[OK] 约束条件添加完成")
print(f"  - 总约束数: {len(prob.constraints)}")

# ============================================================================
# 第四部分：求解MILP模型
# ============================================================================
print("\n[第四部分] 求解MILP模型")
print("-"*80)

print("开始求解MILP模型...")
print("(这可能需要几分钟时间...)")

solver = PULP_CBC_CMD(msg=1, timeLimit=600)
prob.solve(solver)

print(f"\n[OK] 求解完成")
print(f"求解状态: {LpStatus[prob.status]}")

if prob.status == 1:
    optimal_cost = value(prob.objective)
    print(f"最优目标函数值: {optimal_cost:.2f} 元")

    # ============================================================================
    # 第五部分：提取调度方案
    # ============================================================================
    print("\n[第五部分] 提取调度方案")
    print("-"*80)

    schedule = []
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j and value(y[k,i,j]) == 1:
                    num_bikes = int(value(x[k,i,j]))
                    distance = dist[i][j]
                    time = calc_time(i, j, num_bikes)
                    transport_cost_ij = num_bikes * distance * c_transport

                    schedule.append({
                        '卡车编号': k,
                        '起点站点': i,
                        '终点站点': j,
                        '调度量': num_bikes,
                        '距离(km)': distance,
                        '时间(分钟)': time,
                        '运输成本(元)': transport_cost_ij
                    })

    schedule_df = pd.DataFrame(schedule)

    if len(schedule_df) > 0:
        print(f"\n调度方案汇总:")
        print(f"  - 总调度次数: {len(schedule_df)}")
        print(f"  - 总调度量: {schedule_df['调度量'].sum()} 辆")
        print(f"  - 总距离: {schedule_df['距离(km)'].sum():.2f} km")
        print(f"  - 总时间: {schedule_df['时间(分钟)'].sum():.2f} 分钟")
        print(f"  - 运输成本: {schedule_df['运输成本(元)'].sum():.2f} 元")

        print(f"\n详细调度方案:")
        for idx, row in schedule_df.iterrows():
            print(f"  {row['卡车编号']}: {row['起点站点']} → {row['终点站点']}, "
                  f"调度 {row['调度量']} 辆, "
                  f"距离 {row['距离(km)']:.1f}km, "
                  f"时间 {row['时间(分钟)']:.1f}分钟, "
                  f"成本 {row['运输成本(元)']:.2f}元")

        print(f"\n各卡车工作量:")
        for truck in trucks:
            truck_schedule = schedule_df[schedule_df['卡车编号'] == truck]
            if len(truck_schedule) > 0:
                print(f"  {truck}: {len(truck_schedule)}次调度, "
                      f"总时间 {truck_schedule['时间(分钟)'].sum():.1f}分钟, "
                      f"总成本 {truck_schedule['运输成本(元)'].sum():.2f}元")
            else:
                print(f"  {truck}: 未使用")
    else:
        print("\n无需调度（所有站点库存已在期望范围内）")

    # ============================================================================
    # 第六部分：库存对比与成本分析
    # ============================================================================
    print("\n[第六部分] 库存对比与成本分析")
    print("-"*80)

    inventory_comparison = []
    total_shortage_after = 0
    total_excess_after = 0

    for i in stations:
        initial = I0[i]
        expected = I_star[i]
        final = int(value(I[i]))
        allow_dev = delta[i]

        initial_dev = initial - expected
        if initial_dev < -allow_dev:
            initial_shortage = abs(initial_dev) - allow_dev
        else:
            initial_shortage = 0

        if initial_dev > allow_dev:
            initial_excess = initial_dev - allow_dev
        else:
            initial_excess = 0

        final_dev = final - expected
        if final_dev < -allow_dev:
            final_shortage = abs(final_dev) - allow_dev
            total_shortage_after += final_shortage
        else:
            final_shortage = 0

        if final_dev > allow_dev:
            final_excess = final_dev - allow_dev
            total_excess_after += final_excess
        else:
            final_excess = 0

        inventory_comparison.append({
            '站点': i,
            '初始库存': initial,
            '期望库存': expected,
            '允许偏差': allow_dev,
            '调度后库存': final,
            '调度前缺货': initial_shortage,
            '调度前积压': initial_excess,
            '调度后缺货': final_shortage,
            '调度后积压': final_excess
        })

    inventory_df = pd.DataFrame(inventory_comparison)

    print(f"\n成本对比分析:")

    initial_shortage_cost = inventory_df['调度前缺货'].sum() * c_shortage
    initial_excess_cost = inventory_df['调度前积压'].sum() * c_excess
    initial_total_cost = initial_shortage_cost + initial_excess_cost

    print(f"\n调度前:")
    print(f"  - 缺货量: {inventory_df['调度前缺货'].sum():.2f} 辆")
    print(f"  - 积压量: {inventory_df['调度前积压'].sum():.2f} 辆")
    print(f"  - 缺货成本: {initial_shortage_cost:.2f} 元")
    print(f"  - 积压成本: {initial_excess_cost:.2f} 元")
    print(f"  - 总成本: {initial_total_cost:.2f} 元")

    final_shortage_cost = total_shortage_after * c_shortage
    final_excess_cost = total_excess_after * c_excess
    if len(schedule_df) > 0:
        transport_cost_total = schedule_df['运输成本(元)'].sum()
    else:
        transport_cost_total = 0
    final_total_cost = final_shortage_cost + final_excess_cost + transport_cost_total

    print(f"\n调度后:")
    print(f"  - 缺货量: {total_shortage_after:.2f} 辆")
    print(f"  - 积压量: {total_excess_after:.2f} 辆")
    print(f"  - 缺货成本: {final_shortage_cost:.2f} 元")
    print(f"  - 积压成本: {final_excess_cost:.2f} 元")
    print(f"  - 运输成本: {transport_cost_total:.2f} 元")
    print(f"  - 总成本: {final_total_cost:.2f} 元")

    cost_reduction = initial_total_cost - final_total_cost
    if initial_total_cost > 0:
        cost_reduction_pct = (cost_reduction / initial_total_cost) * 100
    else:
        cost_reduction_pct = 0

    print(f"\n经济效益:")
    print(f"  - 成本降低: {cost_reduction:.2f} 元")
    print(f"  - 降低比例: {cost_reduction_pct:.1f}%")

    print(f"\n服务质量指标:")

    stations_in_range_before = sum(1 for _, row in inventory_df.iterrows()
                                    if row['调度前缺货'] == 0 and row['调度前积压'] == 0)
    stations_in_range_after = sum(1 for _, row in inventory_df.iterrows()
                                   if row['调度后缺货'] == 0 and row['调度后积压'] == 0)

    satisfaction_rate_before = (stations_in_range_before / N) * 100
    satisfaction_rate_after = (stations_in_range_after / N) * 100

    print(f"  - 调度前需求满足率: {satisfaction_rate_before:.1f}% ({stations_in_range_before}/{N}个站点)")
    print(f"  - 调度后需求满足率: {satisfaction_rate_after:.1f}% ({stations_in_range_after}/{N}个站点)")
    print(f"  - 满足率提升: {satisfaction_rate_after - satisfaction_rate_before:.1f}%")

    initial_deviations = [abs(row['初始库存'] - row['期望库存']) for _, row in inventory_df.iterrows()]
    final_deviations = [abs(row['调度后库存'] - row['期望库存']) for _, row in inventory_df.iterrows()]

    balance_before = np.std(initial_deviations)
    balance_after = np.std(final_deviations)

    print(f"  - 调度前站点平衡度(标准差): {balance_before:.2f}")
    print(f"  - 调度后站点平衡度(标准差): {balance_after:.2f}")
    print(f"  - 平衡度改善: {((balance_before - balance_after) / balance_before * 100):.1f}%")

    # ============================================================================
    # 第七部分：保存结果
    # ============================================================================
    print("\n[第七部分] 保存结果")
    print("-"*80)

    if len(schedule_df) > 0:
        schedule_df.to_csv('问题2_调度方案.csv', index=False, encoding='utf-8-sig')
        print(f"[OK] 调度方案已保存: 问题2_调度方案.csv")

    inventory_df.to_csv('问题2_库存对比.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] 库存对比已保存: 问题2_库存对比.csv")

    cost_report = {
        '指标': [
            '调度前缺货量(辆)', '调度前积压量(辆)', '调度前缺货成本(元)',
            '调度前积压成本(元)', '调度前总成本(元)',
            '调度后缺货量(辆)', '调度后积压量(辆)', '调度后缺货成本(元)',
            '调度后积压成本(元)', '运输成本(元)', '调度后总成本(元)',
            '成本降低(元)', '成本降低比例(%)',
            '调度前需求满足率(%)', '调度后需求满足率(%)', '满足率提升(%)',
            '调度前平衡度', '调度后平衡度', '平衡度改善(%)',
            '总调度次数', '总调度量(辆)', '总距离(km)', '总时间(分钟)'
        ],
        '数值': [
            inventory_df['调度前缺货'].sum(),
            inventory_df['调度前积压'].sum(),
            initial_shortage_cost,
            initial_excess_cost,
            initial_total_cost,
            total_shortage_after,
            total_excess_after,
            final_shortage_cost,
            final_excess_cost,
            transport_cost_total,
            final_total_cost,
            cost_reduction,
            cost_reduction_pct,
            satisfaction_rate_before,
            satisfaction_rate_after,
            satisfaction_rate_after - satisfaction_rate_before,
            balance_before,
            balance_after,
            (balance_before - balance_after) / balance_before * 100 if balance_before > 0 else 0,
            len(schedule_df) if len(schedule_df) > 0 else 0,
            schedule_df['调度量'].sum() if len(schedule_df) > 0 else 0,
            schedule_df['距离(km)'].sum() if len(schedule_df) > 0 else 0,
            schedule_df['时间(分钟)'].sum() if len(schedule_df) > 0 else 0
        ]
    }

    cost_report_df = pd.DataFrame(cost_report)
    cost_report_df.to_csv('问题2_成本效益分析.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] 成本效益分析已保存: 问题2_成本效益分析.csv")

    summary = {
        '参数/结果': [
            '优化日期', '是否工作日', '站点数', '卡车数', '卡车容量(辆)',
            '时间窗口(分钟)', '平均速度(km/h)', '装卸时间(分钟/辆)',
            '运输成本(元/km·辆)', '缺货成本(元/辆·次)', '积压成本(元/辆·次)',
            '求解状态', '最优目标值(元)', '求解时间(秒)'
        ],
        '值': [
            last_date.strftime('%Y-%m-%d'),
            '是' if is_workday else '否',
            N, M, Q, T_max, v, t_load_per_bike,
            c_transport, c_shortage, c_excess,
            LpStatus[prob.status],
            optimal_cost,
            'N/A'
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('问题2_模型摘要.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] 模型摘要已保存: 问题2_模型摘要.csv")

    # ============================================================================
    # 第八部分：可视化（独立生成每张图）
    # ============================================================================
    print("\n[第八部分] 生成可视化图表")
    print("-"*80)

    # 统一配色方案 - 与问题一保持一致 (Okabe-Ito colorblind-safe调色板)
    OKABE_ITO = {
        'orange': '#E69F00',
        'sky_blue': '#56B4E9', 
        'green': '#009E73',
        'yellow': '#F0E442',
        'blue': '#0072B2',
        'vermillion': '#D55E00',
        'purple': '#CC79A7'
    }

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.unicode_minus': False,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

    sns.set_theme(style='ticks', context='paper', font='Microsoft YaHei', font_scale=1.0)

    # ========================================================================
    # 图1: 站点库存对比
    # ========================================================================
    print("生成图1: 站点库存对比...")
    
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
    print(f"[OK] 图1已保存: 问题2_图1_站点库存对比.png")
    plt.close()

    # ========================================================================
    # 图2: 缺货与积压对比
    # ========================================================================
    print("生成图2: 缺货与积压对比...")
    
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
    print(f"[OK] 图2已保存: 问题2_图2_缺货积压对比.png")
    plt.close()

    # ========================================================================
    # 图3: 成本结构对比
    # ========================================================================
    print("生成图3: 成本结构对比...")
    
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
    print(f"[OK] 图3已保存: 问题2_图3_成本结构对比.png")
    plt.close()

    # ========================================================================
    # 图4: 调度网络示意图（重新设计 - 使用地理坐标和弧线连接）
    # ========================================================================
    print("生成图4: 调度网络示意图...")
    
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
        
        truck_colors = {
            'T1': OKABE_ITO['blue'],
            'T2': OKABE_ITO['orange'],
            'T3': OKABE_ITO['green']
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
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
        
        seen_types = set()
        for stype, scolor in type_colors.items():
            if stype not in seen_types and stype in [c['type'] for c in station_coords.values()]:
                legend_elements.append(Patch(facecolor=scolor, edgecolor='white', 
                                            label=type_labels.get(stype, stype)))
                seen_types.add(stype)
        
        legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=0))
        
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
    print(f"[OK] 图4已保存: 问题2_图4_调度网络示意图.png")
    plt.close()

    # ========================================================================
    # 图5: 各站点偏差改善
    # ========================================================================
    print("生成图5: 各站点偏差改善...")
    
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
    print(f"[OK] 图5已保存: 问题2_图5_各站点偏差改善.png")
    plt.close()

    # ========================================================================
    # 图6: 卡车工作量分布
    # ========================================================================
    print("生成图6: 卡车工作量分布...")
    
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
    print(f"[OK] 图6已保存: 问题2_图6_卡车工作量分布.png")
    plt.close()

    # ========================================================================
    # 图7: 综合性能评估（雷达图）
    # ========================================================================
    print("生成图7: 综合性能评估...")
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    metrics = [
        satisfaction_rate_after,
        100 - (total_shortage_after / max(inventory_df['调度前缺货'].sum(), 1) * 100),
        100 - (total_excess_after / max(inventory_df['调度前积压'].sum(), 1) * 100),
        100 - (transport_cost_total / max(initial_total_cost, 1) * 100),
        100 - (schedule_df['时间(分钟)'].sum() / T_max * 100) if len(schedule_df) > 0 else 100
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
    print(f"[OK] 图7已保存: 问题2_图7_综合性能评估.png")
    plt.close()

    print("\n所有图表生成完成！")

    # ============================================================================
    # 第九部分：参数敏感性分析
    # ============================================================================
    print("\n[第九部分] 参数敏感性分析")
    print("-"*80)

    print("进行成本参数敏感性分析...")

    sensitivity_results = []

    base_params = {
        'c_transport': c_transport,
        'c_shortage': c_shortage,
        'c_excess': c_excess
    }

    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        print(f"  测试运输成本系数: {factor:.2f}x")

        prob_sens = LpProblem("Sensitivity_Transport", LpMinimize)

        x_sens = {}
        y_sens = {}
        for k in trucks:
            for i in stations:
                for j in stations:
                    if i != j:
                        x_sens[k,i,j] = LpVariable(f"xs_{k}_{i}_{j}", lowBound=0, upBound=Q, cat='Integer')
                        y_sens[k,i,j] = LpVariable(f"ys_{k}_{i}_{j}", cat='Binary')

        I_sens = {i: LpVariable(f"Is_{i}", lowBound=0, upBound=C[i], cat='Integer') for i in stations}
        shortage_sens = {i: LpVariable(f"ss_{i}", lowBound=0) for i in stations}
        excess_sens = {i: LpVariable(f"es_{i}", lowBound=0) for i in stations}

        transport_cost_sens = lpSum([x_sens[k,i,j] * dist[i][j] * c_transport * factor
                                     for k in trucks for i in stations for j in stations if i != j])
        shortage_cost_sens = lpSum([shortage_sens[i] * c_shortage for i in stations])
        excess_cost_sens = lpSum([excess_sens[i] * c_excess for i in stations])

        prob_sens += transport_cost_sens + shortage_cost_sens + excess_cost_sens

        for i in stations:
            prob_sens += I_sens[i] == I0[i] + lpSum([x_sens[k,j,i] for k in trucks for j in stations if j != i]) \
                                            - lpSum([x_sens[k,i,j] for k in trucks for j in stations if j != i])
            prob_sens += shortage_sens[i] >= (I_star[i] - delta[i]) - I_sens[i]
            prob_sens += excess_sens[i] >= I_sens[i] - (I_star[i] + delta[i])

        for k in trucks:
            for i in stations:
                for j in stations:
                    if i != j:
                        prob_sens += x_sens[k,i,j] <= Q * y_sens[k,i,j]
                        prob_sens += x_sens[k,i,j] >= y_sens[k,i,j]

        for i in stations:
            prob_sens += lpSum([x_sens[k,i,j] for k in trucks for j in stations if j != i]) <= I0[i]

        for k in trucks:
            avg_bikes = Q / 2
            prob_sens += lpSum([y_sens[k,i,j] * calc_time(i, j, avg_bikes)
                               for i in stations for j in stations if i != j]) <= T_max

        solver_sens = PULP_CBC_CMD(msg=0, timeLimit=60)
        prob_sens.solve(solver_sens)

        if prob_sens.status == 1:
            sensitivity_results.append({
                '参数': f'运输成本×{factor:.2f}',
                '运输成本系数': c_transport * factor,
                '缺货成本系数': c_shortage,
                '积压成本系数': c_excess,
                '最优目标值': value(prob_sens.objective),
                '求解状态': 'Optimal'
            })

    if len(sensitivity_results) > 0:
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df.to_csv('问题2_敏感性分析.csv', index=False, encoding='utf-8-sig')
        print(f"\n[OK] 敏感性分析结果已保存: 问题2_敏感性分析.csv")

        print(f"\n敏感性分析结果:")
        for _, row in sensitivity_df.iterrows():
            print(f"  {row['参数']}: 最优目标值 = {row['最优目标值']:.2f} 元")

else:
    print(f"[ERROR] 模型求解失败，状态: {LpStatus[prob.status]}")
    print("可能原因：")
    print("  1. 问题无可行解（约束条件过于严格）")
    print("  2. 求解时间超限")
    print("  3. 数值不稳定")

print("\n" + "="*80)
print("问题二完整分析完成")
print("="*80)
