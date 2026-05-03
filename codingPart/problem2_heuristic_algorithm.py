"""
问题二：共享单车调度优化（启发式算法）

两阶段启发式算法：
阶段1：贪心匹配 - 识别供给站点和需求站点，进行配对
阶段2：路径优化 - 使用节约算法优化调度路径

作为MILP模型的对照方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("问题二：共享单车调度优化（启发式算法）")
print("="*80)

# ============================================================================
# 第一部分：数据加载
# ============================================================================
print("\n[第一部分] 数据加载")
print("-"*80)

# 读取数据
station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8')
station_features = pd.read_csv('站点特征分析结果.csv', encoding='utf-8')
expected_inventory = pd.read_csv('站点期望库存.csv', encoding='utf-8')
initial_inventory = pd.read_csv('每日初始库存.csv', encoding='utf-8')
distance_matrix = pd.read_csv('站点距离.csv', index_col=0, encoding='utf-8')
weather_data = pd.read_csv('天气数据.csv', encoding='utf-8')

print(f"[OK] 数据加载完成")

# 数据预处理
initial_inventory['日期'] = pd.to_datetime(initial_inventory['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])

# 选择最后一天
last_date = initial_inventory['日期'].max()
weather_info = weather_data[weather_data['日期'] == last_date].iloc[0]
is_workday = 1 - weather_info['是否节假日']

print(f"优化日期: {last_date.strftime('%Y-%m-%d')}")
print(f"是否工作日: {'是' if is_workday else '否'}")

# ============================================================================
# 第二部分：准备参数
# ============================================================================
print("\n[第二部分] 准备参数")
print("-"*80)

stations = station_info['站点ID'].tolist()
N = len(stations)

# 初始库存
last_day_inventory = initial_inventory[initial_inventory['日期'] == last_date]
I0 = {}
for _, row in last_day_inventory.iterrows():
    for station in stations:
        if station in row.index:
            I0[station] = row[station]

# 期望库存
I_star = {}
for _, row in expected_inventory.iterrows():
    station = row['站点ID']
    if is_workday:
        I_star[station] = row['工作日期望库存']
    else:
        I_star[station] = row['周末期望库存']

# 允许偏差
delta = dict(zip(expected_inventory['站点ID'], expected_inventory['允许偏差']))

# 容量
C = dict(zip(station_info['站点ID'], station_info['容量']))

# 距离矩阵
dist = {}
for i in stations:
    dist[i] = {}
    for j in stations:
        dist[i][j] = distance_matrix.loc[i, j]

# 参数
M = 3
Q = 15
v = 20
t_load_per_bike = 1
t_unload_per_bike = 1
T_max = 300
c_transport = 2.0
c_shortage = 10.0
c_excess = 5.0

def calc_time(i, j, num_bikes):
    """计算调度时间"""
    if i == j:
        return 0
    travel_time = (dist[i][j] / v) * 60
    loading_time = num_bikes * t_load_per_bike
    unloading_time = num_bikes * t_unload_per_bike
    return travel_time + loading_time + unloading_time

print(f"站点数: {N}, 卡车数: {M}, 卡车容量: {Q}辆")

# ============================================================================
# 第三部分：阶段1 - 贪心匹配
# ============================================================================
print("\n[第三部分] 阶段1 - 贪心匹配")
print("-"*80)

# 计算每个站点的需求/供给量
supply_demand = {}
for station in stations:
    deviation = I0[station] - I_star[station]
    if deviation < -delta[station]:
        # 需求站点（缺货）
        supply_demand[station] = deviation + delta[station]  # 负值表示需求
    elif deviation > delta[station]:
        # 供给站点（积压）
        supply_demand[station] = deviation - delta[station]  # 正值表示供给
    else:
        # 平衡站点
        supply_demand[station] = 0

# 分类站点
supply_stations = [(s, supply_demand[s]) for s in stations if supply_demand[s] > 0]
demand_stations = [(s, -supply_demand[s]) for s in stations if supply_demand[s] < 0]

supply_stations.sort(key=lambda x: x[1], reverse=True)  # 按供给量降序
demand_stations.sort(key=lambda x: x[1], reverse=True)  # 按需求量降序

print(f"\n供给站点（积压）: {len(supply_stations)}个")
for s, amount in supply_stations:
    print(f"  {s}: 可调出 {amount:.1f}辆")

print(f"\n需求站点（缺货）: {len(demand_stations)}个")
for s, amount in demand_stations:
    print(f"  {s}: 需调入 {amount:.1f}辆")

# 贪心匹配：优先匹配距离近、需求量大的站点对
matches = []
supply_remaining = {s: amount for s, amount in supply_stations}
demand_remaining = {s: amount for s, amount in demand_stations}

# 创建所有可能的配对，按距离排序
all_pairs = []
for supply_s, _ in supply_stations:
    for demand_s, _ in demand_stations:
        all_pairs.append((supply_s, demand_s, dist[supply_s][demand_s]))

all_pairs.sort(key=lambda x: x[2])  # 按距离升序

# 贪心匹配
for supply_s, demand_s, distance in all_pairs:
    if supply_remaining.get(supply_s, 0) > 0 and demand_remaining.get(demand_s, 0) > 0:
        # 计算可调度量（不超过卡车容量）
        transfer_amount = min(
            supply_remaining[supply_s],
            demand_remaining[demand_s],
            Q
        )

        if transfer_amount >= 1:  # 至少调度1辆
            matches.append({
                'from': supply_s,
                'to': demand_s,
                'amount': int(transfer_amount),
                'distance': distance
            })

            supply_remaining[supply_s] -= transfer_amount
            demand_remaining[demand_s] -= transfer_amount

print(f"\n[OK] 贪心匹配完成，生成 {len(matches)} 个调度任务")

# ============================================================================
# 第四部分：阶段2 - 分配给卡车
# ============================================================================
print("\n[第四部分] 阶段2 - 分配给卡车")
print("-"*80)

# 简单策略：按顺序分配给卡车，确保时间约束
trucks = [f'T{i+1}' for i in range(M)]
truck_schedule = {truck: [] for truck in trucks}
truck_time = {truck: 0 for truck in trucks}

for match in matches:
    # 计算该任务的时间
    task_time = calc_time(match['from'], match['to'], match['amount'])

    # 找到当前工作时间最少的卡车
    available_trucks = [(truck, truck_time[truck]) for truck in trucks
                        if truck_time[truck] + task_time <= T_max]

    if available_trucks:
        # 选择工作时间最少的卡车
        selected_truck = min(available_trucks, key=lambda x: x[1])[0]

        truck_schedule[selected_truck].append({
            'from': match['from'],
            'to': match['to'],
            'amount': match['amount'],
            'distance': match['distance'],
            'time': task_time,
            'cost': match['amount'] * match['distance'] * c_transport
        })

        truck_time[selected_truck] += task_time
    else:
        print(f"  警告: 任务 {match['from']}→{match['to']} 无法分配（时间约束）")

print(f"[OK] 任务分配完成")

# 生成最终调度方案
schedule = []
for truck in trucks:
    for task in truck_schedule[truck]:
        schedule.append({
            '卡车编号': truck,
            '起点站点': task['from'],
            '终点站点': task['to'],
            '调度量': task['amount'],
            '距离(km)': task['distance'],
            '时间(分钟)': task['time'],
            '运输成本(元)': task['cost']
        })

schedule_df = pd.DataFrame(schedule)

# ============================================================================
# 第五部分：计算调度后库存和成本
# ============================================================================
print("\n[第五部分] 计算调度后库存和成本")
print("-"*80)

# 计算调度后库存
I_final = I0.copy()
for _, row in schedule_df.iterrows():
    I_final[row['起点站点']] -= row['调度量']
    I_final[row['终点站点']] += row['调度量']

# 计算成本
inventory_comparison = []
total_shortage_after = 0
total_excess_after = 0

for station in stations:
    initial = I0[station]
    expected = I_star[station]
    final = I_final[station]
    allow_dev = delta[station]

    # 调度前
    initial_dev = initial - expected
    initial_shortage = max(0, -initial_dev - allow_dev)
    initial_excess = max(0, initial_dev - allow_dev)

    # 调度后
    final_dev = final - expected
    final_shortage = max(0, -final_dev - allow_dev)
    final_excess = max(0, final_dev - allow_dev)

    total_shortage_after += final_shortage
    total_excess_after += final_excess

    inventory_comparison.append({
        '站点': station,
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

# 成本分析
initial_shortage_cost = inventory_df['调度前缺货'].sum() * c_shortage
initial_excess_cost = inventory_df['调度前积压'].sum() * c_excess
initial_total_cost = initial_shortage_cost + initial_excess_cost

final_shortage_cost = total_shortage_after * c_shortage
final_excess_cost = total_excess_after * c_excess
transport_cost_total = schedule_df['运输成本(元)'].sum() if len(schedule_df) > 0 else 0
final_total_cost = final_shortage_cost + final_excess_cost + transport_cost_total

print(f"\n调度方案汇总:")
print(f"  - 总调度次数: {len(schedule_df)}")
print(f"  - 总调度量: {schedule_df['调度量'].sum()} 辆")
print(f"  - 总距离: {schedule_df['距离(km)'].sum():.2f} km")
print(f"  - 总时间: {schedule_df['时间(分钟)'].sum():.2f} 分钟")
print(f"  - 运输成本: {transport_cost_total:.2f} 元")

print(f"\n成本对比:")
print(f"调度前: {initial_total_cost:.2f}元 (缺货{initial_shortage_cost:.2f} + 积压{initial_excess_cost:.2f})")
print(f"调度后: {final_total_cost:.2f}元 (缺货{final_shortage_cost:.2f} + 积压{final_excess_cost:.2f} + 运输{transport_cost_total:.2f})")
print(f"成本降低: {initial_total_cost - final_total_cost:.2f}元 ({(initial_total_cost - final_total_cost)/initial_total_cost*100:.1f}%)")

# 服务质量
stations_ok_before = sum(1 for _, r in inventory_df.iterrows() if r['调度前缺货']==0 and r['调度前积压']==0)
stations_ok_after = sum(1 for _, r in inventory_df.iterrows() if r['调度后缺货']==0 and r['调度后积压']==0)

print(f"\n服务质量:")
print(f"  - 调度前需求满足率: {stations_ok_before/N*100:.1f}% ({stations_ok_before}/{N})")
print(f"  - 调度后需求满足率: {stations_ok_after/N*100:.1f}% ({stations_ok_after}/{N})")

# ============================================================================
# 第六部分：保存结果
# ============================================================================
print("\n[第六部分] 保存结果")
print("-"*80)

schedule_df.to_csv('问题2_启发式调度方案.csv', index=False, encoding='utf-8-sig')
inventory_df.to_csv('问题2_启发式库存对比.csv', index=False, encoding='utf-8-sig')

# 保存成本报告
cost_report = {
    '指标': [
        '调度前总成本(元)', '调度后总成本(元)', '成本降低(元)', '成本降低比例(%)',
        '运输成本(元)', '总调度次数', '总调度量(辆)', '总距离(km)', '总时间(分钟)',
        '调度前需求满足率(%)', '调度后需求满足率(%)'
    ],
    '数值': [
        initial_total_cost, final_total_cost, initial_total_cost - final_total_cost,
        (initial_total_cost - final_total_cost)/initial_total_cost*100 if initial_total_cost > 0 else 0,
        transport_cost_total, len(schedule_df), schedule_df['调度量'].sum(),
        schedule_df['距离(km)'].sum(), schedule_df['时间(分钟)'].sum(),
        stations_ok_before/N*100, stations_ok_after/N*100
    ]
}

cost_report_df = pd.DataFrame(cost_report)
cost_report_df.to_csv('问题2_启发式成本分析.csv', index=False, encoding='utf-8-sig')

print(f"[OK] 结果已保存:")
print(f"  - 问题2_启发式调度方案.csv")
print(f"  - 问题2_启发式库存对比.csv")
print(f"  - 问题2_启发式成本分析.csv")

print("\n" + "="*80)
print("启发式算法完成")
print("="*80)
