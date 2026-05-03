"""
问题三：基于XGBoost需求预测的调度优化（最终版）
功能：
1. 使用XGBoost预测未来需求
2. 基于预测需求生成调度方案
3. 对比"预测需求"vs"历史平均需求"的调度效果
"""

import pandas as pd
import numpy as np
import pickle
from pulp import *
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("问题三：基于XGBoost需求预测的调度优化")
print("="*80)

# ============================================================================
# 1. 加载预测结果
# ============================================================================

print("\n[步骤1] 加载XGBoost预测结果...")

# 加载XGBoost预测结果
xgb_results = pd.read_csv('问题3_XGBoost预测结果.csv', encoding='utf-8-sig')
print(f"  - XGBoost预测结果: {len(xgb_results)} 条")

# 加载XGBoost模型
with open('问题3_XGBoost模型.pkl', 'rb') as f:
    xgb_model_data = pickle.load(f)

print(f"\n  XGBoost模型性能:")
print(f"    训练集 MAE:  {xgb_model_data['performance']['train_mae']:.2f}")
print(f"    测试集 MAE:  {xgb_model_data['performance']['test_mae']:.2f}")
print(f"    测试集 MAPE: {xgb_model_data['performance']['test_mape']:.2f}%")
print(f"    测试集 R2:   {xgb_model_data['performance']['test_r2']:.4f}")

# 加载基础数据
station_info = pd.read_csv('../modelingPart/站点基础信息.csv', encoding='utf-8-sig')
distance_matrix = pd.read_csv('../modelingPart/站点距离.csv', index_col=0, encoding='utf-8-sig')
inventory_data = pd.read_csv('../modelingPart/每日初始库存.csv', encoding='utf-8-sig')
target_inventory = pd.read_csv('../modelingPart/站点期望库存.csv', encoding='utf-8-sig')

print(f"  - 站点信息: {len(station_info)} 个站点")

# ============================================================================
# 2. 定义MILP调度优化函数
# ============================================================================

print("\n[步骤2] 定义MILP调度优化函数...")

def optimize_schedule(I0_dict, I_star_dict, station_info, distance_matrix):
    """
    MILP调度优化

    参数:
        I0_dict: 初始库存字典 {station_id: inventory}
        I_star_dict: 期望库存字典 {station_id: target}
        station_info: 站点信息DataFrame
        distance_matrix: 距离矩阵DataFrame

    返回:
        schedule: 调度方案列表
        total_cost: 总成本
        final_inventory: 调度后库存
        cost_breakdown: 成本分解
    """

    # 参数设置
    stations = list(I0_dict.keys())
    N = len(stations)
    M = 3  # 卡车数量
    trucks = [f'T{i+1}' for i in range(M)]
    Q = 15  # 卡车容量

    # 站点容量
    C = dict(zip(station_info['station_id'], station_info['容量']))

    # 距离矩阵
    dist = {}
    for i in stations:
        dist[i] = {}
        for j in stations:
            dist[i][j] = distance_matrix.loc[i, j]

    # 成本参数
    c_transport = 2.0  # 运输成本
    c_shortage = 10.0  # 缺货成本
    c_excess = 5.0     # 积压成本

    # 允许偏差（简化：统一设为3）
    delta = {s: 3 for s in stations}

    # 时间参数
    v = 20  # 速度 km/h
    t_load = 1  # 装载时间 分钟/辆
    t_unload = 1  # 卸载时间 分钟/辆
    T_max = 300  # 时间窗口 分钟

    def calc_time(i, j, num_bikes):
        if i == j:
            return 0
        travel_time = (dist[i][j] / v) * 60
        loading_time = num_bikes * t_load
        unloading_time = num_bikes * t_unload
        return travel_time + loading_time + unloading_time

    # 构建MILP模型
    prob = LpProblem("Bike_Scheduling", LpMinimize)

    # 决策变量
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

    # 目标函数
    transport_cost = lpSum([x[k,i,j] * dist[i][j] * c_transport
                            for k in trucks for i in stations for j in stations if i != j])
    shortage_cost = lpSum([shortage_var[i] * c_shortage for i in stations])
    excess_cost = lpSum([excess_var[i] * c_excess for i in stations])

    prob += transport_cost + shortage_cost + excess_cost

    # 约束条件
    # 1. 库存平衡
    for i in stations:
        prob += I[i] == I0_dict[i] + lpSum([x[k,j,i] for k in trucks for j in stations if j != i]) \
                              - lpSum([x[k,i,j] for k in trucks for j in stations if j != i])

    # 2. 缺货和积压定义
    for i in stations:
        prob += shortage_var[i] >= (I_star_dict[i] - delta[i]) - I[i]
        prob += excess_var[i] >= I[i] - (I_star_dict[i] + delta[i])

    # 3. 容量约束
    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j:
                    prob += x[k,i,j] <= Q * y[k,i,j]
                    prob += x[k,i,j] >= y[k,i,j]

    # 4. 可行性约束
    for i in stations:
        prob += lpSum([x[k,i,j] for k in trucks for j in stations if j != i]) <= I0_dict[i]

    # 5. 时间窗口约束
    for k in trucks:
        avg_bikes = Q / 2
        prob += lpSum([y[k,i,j] * calc_time(i, j, avg_bikes)
                       for i in stations for j in stations if i != j]) <= T_max

    # 求解
    solver = PULP_CBC_CMD(msg=0, timeLimit=300)
    prob.solve(solver)

    # 提取结果
    schedule = []
    cost_breakdown = {'transport': 0, 'shortage': 0, 'excess': 0}

    if prob.status == 1:
        for k in trucks:
            for i in stations:
                for j in stations:
                    if i != j and value(y[k,i,j]) == 1:
                        num_bikes = int(value(x[k,i,j]))
                        if num_bikes > 0:
                            transport_cost_val = num_bikes * dist[i][j] * c_transport
                            schedule.append({
                                '卡车编号': k,
                                '起点站点': i,
                                '终点站点': j,
                                '调度量': num_bikes,
                                '距离': dist[i][j],
                                '运输成本': transport_cost_val
                            })
                            cost_breakdown['transport'] += transport_cost_val

        # 计算偏差成本
        for i in stations:
            shortage_val = value(shortage_var[i])
            excess_val = value(excess_var[i])
            if shortage_val > 0:
                cost_breakdown['shortage'] += shortage_val * c_shortage
            if excess_val > 0:
                cost_breakdown['excess'] += excess_val * c_excess

        total_cost = value(prob.objective)
        final_inventory = {i: int(value(I[i])) for i in stations}

        return schedule, total_cost, final_inventory, cost_breakdown
    else:
        return [], None, None, None

print("  [OK] MILP优化函数定义完成")

# ============================================================================
# 3. 方案A：基于历史平均需求的调度
# ============================================================================

print("\n[步骤3] 方案A：基于历史平均需求的调度...")

# 站点列表
stations = ['S001', 'S002', 'S003', 'S004', 'S005',
            'S006', 'S007', 'S008', 'S009', 'S010']

# 计算历史平均期望库存（前10天）
historical_avg = {}
for station in stations:
    target_row = target_inventory[target_inventory['站点ID'] == station].iloc[0]
    historical_avg[station] = (target_row['工作日期望库存'] + target_row['周末期望库存']) / 2

print("  历史平均期望库存:")
for station in stations:
    print(f"    {station}: {historical_avg[station]:.1f} 辆")

# 使用第11天的初始库存作为起点
day11_inventory = inventory_data[inventory_data['日期'] == '2023/6/11']
I0_day11 = {station: int(day11_inventory[station].values[0]) for station in stations}

print(f"\n  第11天初始库存:")
for station in stations:
    print(f"    {station}: {I0_day11[station]} 辆")

# 执行调度优化（方案A）
print(f"\n  执行MILP优化（方案A）...")
schedule_A, cost_A, inventory_A, breakdown_A = optimize_schedule(
    I0_day11, historical_avg, station_info, distance_matrix
)

if cost_A is not None:
    print(f"  [OK] 方案A优化完成")
    print(f"    调度方案数: {len(schedule_A)} 条")
    print(f"    总成本: {cost_A:.2f} 元")
    print(f"      - 运输成本: {breakdown_A['transport']:.2f} 元")
    print(f"      - 缺货成本: {breakdown_A['shortage']:.2f} 元")
    print(f"      - 积压成本: {breakdown_A['excess']:.2f} 元")
else:
    print(f"  [ERROR] 方案A优化失败")

# ============================================================================
# 4. 方案B：基于XGBoost预测需求的调度
# ============================================================================

print("\n[步骤4] 方案B：基于XGBoost预测需求的调度...")

# 使用XGBoost预测的第11天期望库存
day11_predictions = xgb_results[xgb_results['date'] == '2023-06-11']
I_star_predicted = {row['station_id']: row['predicted']
                    for _, row in day11_predictions.iterrows()}

print("  XGBoost预测期望库存（第11天）:")
for station in stations:
    print(f"    {station}: {I_star_predicted[station]:.1f} 辆")

# 执行调度优化（方案B）
print(f"\n  执行MILP优化（方案B）...")
schedule_B, cost_B, inventory_B, breakdown_B = optimize_schedule(
    I0_day11, I_star_predicted, station_info, distance_matrix
)

if cost_B is not None:
    print(f"  [OK] 方案B优化完成")
    print(f"    调度方案数: {len(schedule_B)} 条")
    print(f"    总成本: {cost_B:.2f} 元")
    print(f"      - 运输成本: {breakdown_B['transport']:.2f} 元")
    print(f"      - 缺货成本: {breakdown_B['shortage']:.2f} 元")
    print(f"      - 积压成本: {breakdown_B['excess']:.2f} 元")
else:
    print(f"  [ERROR] 方案B优化失败")

# ============================================================================
# 5. 对比分析
# ============================================================================

print("\n[步骤5] 对比分析...")

if cost_A is not None and cost_B is not None:
    # 成本对比
    cost_reduction = cost_A - cost_B
    cost_reduction_pct = (cost_reduction / cost_A) * 100

    print(f"\n  成本对比:")
    print(f"    方案A（历史平均）: {cost_A:.2f} 元")
    print(f"    方案B（XGBoost预测）: {cost_B:.2f} 元")
    print(f"    成本降低: {cost_reduction:.2f} 元 ({cost_reduction_pct:.1f}%)")

    # 库存偏差对比
    actual_target_day11 = {row['station_id']: row['target']
                          for _, row in day11_predictions.iterrows()}

    deviation_A = sum([abs(inventory_A[s] - actual_target_day11[s]) for s in stations])
    deviation_B = sum([abs(inventory_B[s] - actual_target_day11[s]) for s in stations])

    deviation_reduction = deviation_A - deviation_B
    deviation_reduction_pct = (deviation_reduction / deviation_A) * 100 if deviation_A > 0 else 0

    print(f"\n  库存偏差对比:")
    print(f"    方案A总偏差: {deviation_A:.1f} 辆")
    print(f"    方案B总偏差: {deviation_B:.1f} 辆")
    print(f"    偏差降低: {deviation_reduction:.1f} 辆 ({deviation_reduction_pct:.1f}%)")

    # 调度效率对比
    total_bikes_A = sum([s['调度量'] for s in schedule_A])
    total_bikes_B = sum([s['调度量'] for s in schedule_B])

    print(f"\n  调度效率对比:")
    print(f"    方案A调度总量: {total_bikes_A} 辆")
    print(f"    方案B调度总量: {total_bikes_B} 辆")
    print(f"    方案A调度方案数: {len(schedule_A)} 条")
    print(f"    方案B调度方案数: {len(schedule_B)} 条")

    # 保存对比结果
    comparison_df = pd.DataFrame({
        '指标': ['总成本（元）', '运输成本（元）', '偏差成本（元）',
                '库存总偏差（辆）', '调度总量（辆）', '调度方案数'],
        '方案A（历史平均）': [
            cost_A,
            breakdown_A['transport'],
            breakdown_A['shortage'] + breakdown_A['excess'],
            deviation_A,
            total_bikes_A,
            len(schedule_A)
        ],
        '方案B（XGBoost预测）': [
            cost_B,
            breakdown_B['transport'],
            breakdown_B['shortage'] + breakdown_B['excess'],
            deviation_B,
            total_bikes_B,
            len(schedule_B)
        ],
        '改进幅度': [
            f'{cost_reduction_pct:.1f}%',
            f'{(breakdown_A["transport"] - breakdown_B["transport"])/breakdown_A["transport"]*100:.1f}%',
            f'{((breakdown_A["shortage"]+breakdown_A["excess"]) - (breakdown_B["shortage"]+breakdown_B["excess"]))/(breakdown_A["shortage"]+breakdown_A["excess"])*100:.1f}%',
            f'{deviation_reduction_pct:.1f}%',
            f'{total_bikes_A - total_bikes_B}',
            f'{len(schedule_A) - len(schedule_B)}'
        ]
    })

    comparison_df.to_csv('问题3_最终方案对比.csv', index=False, encoding='utf-8-sig')

    # 保存调度方案
    if len(schedule_A) > 0:
        pd.DataFrame(schedule_A).to_csv('问题3_最终方案A调度方案.csv', index=False, encoding='utf-8-sig')
    if len(schedule_B) > 0:
        pd.DataFrame(schedule_B).to_csv('问题3_最终方案B调度方案.csv', index=False, encoding='utf-8-sig')

    print(f"\n[完成] 对比结果已保存:")
    print(f"  - 问题3_最终方案对比.csv")
    print(f"  - 问题3_最终方案A调度方案.csv")
    print(f"  - 问题3_最终方案B调度方案.csv")

print("\n" + "="*80)
print("基于XGBoost需求预测的调度优化完成！")
print("="*80)
