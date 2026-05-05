"""
问题二：共享单车调度优化 - 模型部分
功能：数据加载、MILP优化模型、求解、保存结果

输出文件：
  - 问题2_调度方案.csv
  - 问题2_库存对比.csv
  - 问题2_成本效益分析.csv
  - 问题2_模型摘要.csv

使用方法：
  python problem2_model.py
"""

import pandas as pd
import numpy as np
from pulp import *
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("问题二：共享单车调度优化（MILP模型）")
print("="*80)

# ============================================================================
# 第一部分：数据加载与预处理
# ============================================================================
print("\n[第一部分] 数据加载与预处理")
print("-"*80)

station_info = pd.read_csv('附件数据/站点基础信息.csv', encoding='utf-8')
station_features = pd.read_csv('站点特征分析结果.csv', encoding='utf-8')
expected_inventory = pd.read_csv('附件数据/站点期望库存.csv', encoding='utf-8')
initial_inventory = pd.read_csv('附件数据/每日初始库存.csv', encoding='utf-8')
distance_matrix = pd.read_csv('附件数据/站点距离.csv', index_col=0, encoding='utf-8')
weather_data = pd.read_csv('附件数据/天气数据.csv', encoding='utf-8')

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
    # 第八部分：参数敏感性分析
    # ============================================================================
    print("\n[第八部分] 参数敏感性分析")
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
        shortage_sens = {i: LpVariable(f"shortages_{i}", lowBound=0) for i in stations}
        excess_sens = {i: LpVariable(f"excesss_{i}", lowBound=0) for i in stations}

        c_trans_sens = c_transport * factor

        prob_sens += (lpSum([x_sens[k,i,j] * dist[i][j] * c_trans_sens
                             for k in trucks for i in stations for j in stations if i != j])
                     + lpSum([shortage_sens[i] * c_shortage for i in stations])
                     + lpSum([excess_sens[i] * c_excess for i in stations])), "Total_Cost_Sens"

        for i in stations:
            prob_sens += I_sens[i] == I0[i] + lpSum([x_sens[k,j,i] for k in trucks for j in stations if j != i]) \
                                  - lpSum([x_sens[k,i,j] for k in trucks for j in stations if j != i]), f"Inv_Balance_S_{i}"

        for i in stations:
            prob_sens += shortage_sens[i] >= (I_star[i] - delta[i]) - I_sens[i], f"Shortage_S_{i}"
            prob_sens += excess_sens[i] >= I_sens[i] - (I_star[i] + delta[i]), f"Excess_S_{i}"

        for k in trucks:
            for i in stations:
                for j in stations:
                    if i != j:
                        prob_sens += x_sens[k,i,j] <= Q * y_sens[k,i,j], f"Cap_S_{k}_{i}_{j}"

        for k in trucks:
            for i in stations:
                for j in stations:
                    if i != j:
                        prob_sens += x_sens[k,i,j] >= y_sens[k,i,j], f"Logic_S_{k}_{i}_{j}"

        for i in stations:
            prob_sens += lpSum([x_sens[k,i,j] for k in trucks for j in stations if j != i]) <= I0[i], f"Feas_S_{i}"

        for k in trucks:
            avg_bikes = Q / 2
            prob_sens += lpSum([y_sens[k,i,j] * calc_time(i, j, avg_bikes)
                                for i in stations for j in stations if i != j]) <= T_max, f"Time_S_{k}"

        solver_sens = PULP_CBC_CMD(msg=0, timeLimit=300)
        prob_sens.solve(solver_sens)

        if prob_sens.status == 1:
            sens_cost = value(prob_sens.objective)
            sensitivity_results.append({
                '运输成本系数': f'{factor:.2f}x',
                '最优目标值(元)': sens_cost
            })
            print(f"    最优目标值 = {sens_cost:.2f} 元")
        else:
            print(f"    未找到可行解")

    if len(sensitivity_results) > 0:
        sens_df = pd.DataFrame(sensitivity_results)
        sens_df.to_csv('问题2_敏感性分析.csv', index=False, encoding='utf-8-sig')
        print(f"\n[OK] 敏感性分析结果已保存: 问题2_敏感性分析.csv")

        print(f"\n敏感性分析结果:")
        for _, row in sens_df.iterrows():
            print(f"  运输成本{row['运输成本系数']}: 最优目标值 = {row['最优目标值(元)']:.2f} 元")

    print("\n" + "="*80)
    print("问题二模型求解完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  - 问题2_调度方案.csv")
    print("  - 问题2_库存对比.csv")
    print("  - 问题2_成本效益分析.csv")
    print("  - 问题2_模型摘要.csv")
    print("  - 问题2_敏感性分析.csv")
    print("\n下一步: 运行 python problem2_visualization.py 生成可视化图表")

else:
    print("\n[ERROR] 模型未找到最优解！请检查约束条件或数据。")