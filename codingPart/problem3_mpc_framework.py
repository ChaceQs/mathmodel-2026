"""
问题三：滚动时域预测-调度一体化框架（MPC）
实现模型预测控制（Model Predictive Control）框架，提升系统鲁棒性

核心思想：
1. 每天根据最新数据预测未来需求
2. 优化生成调度方案
3. 执行当天方案
4. 观测实际结果并更新模型
5. 滚动进行下一天
"""

import pandas as pd
import numpy as np
import pickle
from pulp import *
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("问题三：滚动时域预测-调度一体化框架（MPC）")
print("="*80)

# ============================================================================
# 1. 加载数据和模型
# ============================================================================

print("\n[步骤1] 加载数据和模型...")

# 加载XGBoost模型
with open('问题3_XGBoost模型.pkl', 'rb') as f:
    xgb_model_data = pickle.load(f)
    xgb_model = xgb_model_data['model']
    feature_cols = xgb_model_data['feature_cols']

print(f"  - XGBoost模型加载完成")
print(f"  - 特征维度: {len(feature_cols)}")

# 加载基础数据
riding_data = pd.read_csv('日汇总骑行数据.csv', encoding='utf-8-sig')
inventory_data = pd.read_csv('每日初始库存.csv', encoding='utf-8-sig')
weather_data = pd.read_csv('天气数据.csv', encoding='utf-8-sig')
target_inventory = pd.read_csv('站点期望库存.csv', encoding='utf-8-sig')
station_info = pd.read_csv('站点基础信息.csv', encoding='utf-8-sig')
distance_matrix = pd.read_csv('站点距离.csv', index_col=0, encoding='utf-8-sig')

# 转换日期格式
riding_data['日期'] = pd.to_datetime(riding_data['日期'])
inventory_data['日期'] = pd.to_datetime(inventory_data['日期'])
weather_data['日期'] = pd.to_datetime(weather_data['日期'])

stations = ['S001', 'S002', 'S003', 'S004', 'S005',
            'S006', 'S007', 'S008', 'S009', 'S010']

print(f"  - 历史数据: {len(inventory_data)} 天")
print(f"  - 站点数量: {len(stations)} 个")

# ============================================================================
# 2. 定义特征生成函数
# ============================================================================

print("\n[步骤2] 定义特征生成函数...")

def create_features_for_day(date, station_id, historical_data):
    """
    为指定日期和站点生成预测特征

    参数:
        date: 预测日期
        station_id: 站点ID
        historical_data: 历史数据字典

    返回:
        features: 特征字典
    """
    # 获取站点信息
    station_row = station_info[station_info['station_id'] == station_id].iloc[0]
    target_row = target_inventory[target_inventory['站点ID'] == station_id].iloc[0]

    # 站点类型编码
    type_mapping = {'居民区': 0, 'residential': 0,
                   '商务区': 1, 'business': 1,
                   '教育区': 2, 'education': 2,
                   '交通枢纽': 3, 'transport': 3}
    station_type_code = type_mapping.get(station_row['类型'], 0)

    # 获取天气数据
    weather_row = weather_data[weather_data['日期'] == date]
    if len(weather_row) == 0:
        # 如果没有天气数据，使用默认值
        day_of_week = date.dayofweek + 1
        is_weekend = 1 if day_of_week in [6, 7] else 0
        temperature = 26.0
        weather_coef = 1.0
    else:
        weather_row = weather_row.iloc[0]
        day_of_week = weather_row['星期']
        is_weekend = 1 if day_of_week in [6, 7] else 0
        temperature = weather_row['平均温度']
        weather_coef = weather_row['天气系数']

    # 获取当前库存
    current_inventory = historical_data['inventory'].get(station_id, 25)

    # 获取当前骑行数据
    riding_row = riding_data[(riding_data['日期'] == date) &
                            (riding_data['站点 ID'] == station_id)]
    if len(riding_row) > 0:
        riding_row = riding_row.iloc[0]
        total_outflow = riding_row['总流出']
        total_inflow = riding_row['总流入']
        net_flow = riding_row['净流量']
        morning_outflow = riding_row['早高峰流出']
        morning_inflow = riding_row['早高峰流入']
        evening_outflow = riding_row['晚高峰流出']
        evening_inflow = riding_row['晚高峰流入']
    else:
        # 使用历史平均值
        total_outflow = 35
        total_inflow = 35
        net_flow = 0
        morning_outflow = 18
        morning_inflow = 18
        evening_outflow = 17
        evening_inflow = 17

    # 基础特征
    features = {
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_holiday': 0,  # 简化
        'day_of_month': date.day,
        'station_type': station_type_code,
        'station_capacity': station_row['容量'],
        'target_weekday': target_row['工作日期望库存'],
        'target_weekend': target_row['周末期望库存'],
        'temperature': temperature,
        'weather_coef': weather_coef,
        'current_inventory': current_inventory,
        'total_outflow': total_outflow,
        'total_inflow': total_inflow,
        'net_flow': net_flow,
        'morning_outflow': morning_outflow,
        'morning_inflow': morning_inflow,
        'evening_outflow': evening_outflow,
        'evening_inflow': evening_inflow,
    }

    # 滞后特征
    lag1_inv = historical_data['inventory_history'].get((station_id, -1), current_inventory)
    lag7_inv = historical_data['inventory_history'].get((station_id, -7), current_inventory)

    features['inventory_lag1'] = lag1_inv
    features['inventory_lag7'] = lag7_inv
    features['net_flow_lag1'] = 0  # 简化

    # 滚动统计特征
    recent_invs = [historical_data['inventory_history'].get((station_id, -i), current_inventory)
                   for i in range(1, 4)]
    features['inventory_rolling_mean_3'] = np.mean(recent_invs)
    features['inventory_rolling_std_3'] = np.std(recent_invs)

    # 交互特征
    features['inventory_x_weekend'] = current_inventory * is_weekend
    features['type_x_dayofweek'] = station_type_code * day_of_week
    features['netflow_x_weather'] = net_flow * weather_coef

    return features

print("  [OK] 特征生成函数定义完成")

# ============================================================================
# 3. 定义MILP调度优化函数
# ============================================================================

print("\n[步骤3] 定义MILP调度优化函数...")

def optimize_schedule_mpc(I0_dict, I_star_dict, station_info, distance_matrix, verbose=False):
    """
    MILP调度优化（MPC版本）
    """
    stations = list(I0_dict.keys())
    M = 3
    trucks = [f'T{i+1}' for i in range(M)]
    Q = 15

    C = dict(zip(station_info['station_id'], station_info['容量']))

    dist = {}
    for i in stations:
        dist[i] = {}
        for j in stations:
            dist[i][j] = distance_matrix.loc[i, j]

    c_transport = 2.0
    c_shortage = 10.0
    c_excess = 5.0
    delta = {s: 3 for s in stations}

    v = 20
    t_load = 1
    t_unload = 1
    T_max = 300

    def calc_time(i, j, num_bikes):
        if i == j:
            return 0
        return (dist[i][j] / v) * 60 + num_bikes * (t_load + t_unload)

    prob = LpProblem("MPC_Scheduling", LpMinimize)

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

    transport_cost = lpSum([x[k,i,j] * dist[i][j] * c_transport
                            for k in trucks for i in stations for j in stations if i != j])
    shortage_cost = lpSum([shortage_var[i] * c_shortage for i in stations])
    excess_cost = lpSum([excess_var[i] * c_excess for i in stations])

    prob += transport_cost + shortage_cost + excess_cost

    for i in stations:
        prob += I[i] == I0_dict[i] + lpSum([x[k,j,i] for k in trucks for j in stations if j != i]) \
                              - lpSum([x[k,i,j] for k in trucks for j in stations if j != i])

    for i in stations:
        prob += shortage_var[i] >= (I_star_dict[i] - delta[i]) - I[i]
        prob += excess_var[i] >= I[i] - (I_star_dict[i] + delta[i])

    for k in trucks:
        for i in stations:
            for j in stations:
                if i != j:
                    prob += x[k,i,j] <= Q * y[k,i,j]
                    prob += x[k,i,j] >= y[k,i,j]

    for i in stations:
        prob += lpSum([x[k,i,j] for k in trucks for j in stations if j != i]) <= I0_dict[i]

    for k in trucks:
        avg_bikes = Q / 2
        prob += lpSum([y[k,i,j] * calc_time(i, j, avg_bikes)
                       for i in stations for j in stations if i != j]) <= T_max

    solver = PULP_CBC_CMD(msg=0, timeLimit=180)
    prob.solve(solver)

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
                                '调度量': num_bikes
                            })
                            cost_breakdown['transport'] += transport_cost_val

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
# 4. 滚动时域MPC框架
# ============================================================================

print("\n[步骤4] 执行滚动时域MPC框架...")
print("\n" + "="*80)
print("开始滚动优化（第11-14天）")
print("="*80)

# 初始化历史数据
historical_data = {
    'inventory': {},
    'inventory_history': {}
}

# 加载前10天的库存历史
for day_idx in range(10):
    date = pd.Timestamp('2023-06-01') + pd.Timedelta(days=day_idx)
    inv_row = inventory_data[inventory_data['日期'] == date]
    if len(inv_row) > 0:
        for station in stations:
            historical_data['inventory_history'][(station, day_idx - 9)] = inv_row[station].values[0]

# 第10天的库存作为初始状态
day10_inv = inventory_data[inventory_data['日期'] == '2023-06-10']
for station in stations:
    historical_data['inventory'][station] = day10_inv[station].values[0]

# MPC循环
mpc_results = []
total_cost_mpc = 0

for day_offset in range(4):  # 第11-14天
    current_day = 11 + day_offset
    current_date = pd.Timestamp('2023-06-01') + pd.Timedelta(days=current_day - 1)

    print(f"\n{'─'*80}")
    print(f"第{current_day}天 ({current_date.strftime('%Y-%m-%d')})")
    print(f"{'─'*80}")

    # 步骤1：预测当天需求
    print(f"\n  [1] 预测需求...")
    predictions = {}

    for station in stations:
        # 生成特征
        features = create_features_for_day(current_date, station, historical_data)

        # 确保特征顺序与训练时一致
        X_pred = pd.DataFrame([features])[feature_cols]

        # 预测
        pred = xgb_model.predict(X_pred)[0]
        predictions[station] = pred

        print(f"    {station}: {pred:.1f} 辆")

    # 步骤2：优化调度方案
    print(f"\n  [2] 优化调度方案...")
    I0_current = historical_data['inventory'].copy()

    schedule, cost, final_inv, breakdown = optimize_schedule_mpc(
        I0_current, predictions, station_info, distance_matrix
    )

    if cost is not None:
        print(f"    调度方案: {len(schedule)} 条")
        print(f"    预计成本: {cost:.2f} 元")
        total_cost_mpc += cost

        # 显示调度方案
        if len(schedule) > 0:
            print(f"\n    调度详情:")
            for s in schedule:
                print(f"      {s['卡车编号']}: {s['起点站点']} → {s['终点站点']} ({s['调度量']}辆)")
    else:
        print(f"    [警告] 优化失败")
        final_inv = I0_current

    # 步骤3：执行调度（模拟）
    print(f"\n  [3] 执行调度并观测结果...")

    # 获取实际库存（如果有的话）
    actual_inv_row = inventory_data[inventory_data['日期'] == current_date]
    if len(actual_inv_row) > 0:
        actual_inv = {station: actual_inv_row[station].values[0] for station in stations}
        print(f"    实际库存已观测")
    else:
        # 使用优化后的库存作为"实际"库存
        actual_inv = final_inv
        print(f"    使用优化库存作为实际库存")

    # 步骤4：更新历史数据
    print(f"\n  [4] 更新模型状态...")
    for station in stations:
        # 更新当前库存
        historical_data['inventory'][station] = actual_inv[station]

        # 更新历史记录
        historical_data['inventory_history'][(station, -1)] = actual_inv[station]

    print(f"    历史数据已更新")

    # 记录结果
    mpc_results.append({
        '日期': current_date,
        '调度方案数': len(schedule),
        '成本': cost if cost is not None else 0,
        '预测': predictions,
        '实际库存': actual_inv
    })

# ============================================================================
# 5. MPC结果汇总
# ============================================================================

print(f"\n{'='*80}")
print("滚动时域MPC结果汇总")
print(f"{'='*80}")

print(f"\n总成本: {total_cost_mpc:.2f} 元")
print(f"平均每天成本: {total_cost_mpc/4:.2f} 元")

# 保存MPC结果
mpc_summary = pd.DataFrame([{
    '日期': r['日期'].strftime('%Y-%m-%d'),
    '调度方案数': r['调度方案数'],
    '成本': r['成本']
} for r in mpc_results])

mpc_summary.to_csv('问题3_MPC滚动优化结果.csv', index=False, encoding='utf-8-sig')

print(f"\n[完成] MPC结果已保存:")
print(f"  - 问题3_MPC滚动优化结果.csv")

# ============================================================================
# 6. 与静态方案对比
# ============================================================================

print(f"\n{'='*80}")
print("MPC vs 静态方案对比")
print(f"{'='*80}")

# 读取之前的静态优化结果
try:
    static_comparison = pd.read_csv('问题3_最终方案对比.csv', encoding='utf-8-sig')
    static_cost_B = float(static_comparison[static_comparison['指标'] == '总成本（元）']['方案B（XGBoost预测）'].values[0])

    print(f"\n静态优化（单次预测）:")
    print(f"  第11天成本: {static_cost_B:.2f} 元")

    print(f"\nMPC滚动优化（4天）:")
    print(f"  总成本: {total_cost_mpc:.2f} 元")
    print(f"  平均每天: {total_cost_mpc/4:.2f} 元")

    print(f"\nMPC优势:")
    print(f"  - 实时反馈：每天根据最新数据调整")
    print(f"  - 鲁棒性强：适应实际变化")
    print(f"  - 持续优化：滚动更新模型状态")

except:
    print(f"\n[提示] 未找到静态方案对比数据")

print(f"\n{'='*80}")
print("滚动时域预测-调度一体化框架完成！")
print(f"{'='*80}")
