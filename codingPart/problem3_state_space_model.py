"""
问题三：状态空间模型（卡尔曼滤波）
基于系统动力学的需求预测，提供物理约束
"""

import pandas as pd
import numpy as np
from scipy.linalg import solve_discrete_are
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("问题三：状态空间模型（卡尔曼滤波）")
print("="*80)

# ============================================================================
# 1. 数据加载
# ============================================================================

print("\n[步骤1] 加载数据...")

riding_data = pd.read_csv('../modelingPart/日汇总骑行数据.csv', encoding='utf-8-sig')
inventory_data = pd.read_csv('../modelingPart/每日初始库存.csv', encoding='utf-8-sig')
target_inventory = pd.read_csv('../modelingPart/站点期望库存.csv', encoding='utf-8-sig')

print(f"  - 骑行数据: {len(riding_data)} 条")
print(f"  - 库存数据: {len(inventory_data)} 天")

# ============================================================================
# 2. 参数估计
# ============================================================================

print("\n[步骤2] 估计状态空间模型参数...")

# 站点列表
stations = ['S001', 'S002', 'S003', 'S004', 'S005',
            'S006', 'S007', 'S008', 'S009', 'S010']
n_stations = len(stations)

# 转移率矩阵A（站点间流动关系）
print("  - 估计转移率矩阵A...")

# 计算每个站点的平均净流量
avg_net_flow = {}
for station in stations:
    station_data = riding_data[riding_data['站点 ID'] == station]
    avg_net_flow[station] = station_data['净流量'].mean()

# 构建转移率矩阵（简化版：对角矩阵 + 小扰动）
A = np.eye(n_stations) * 0.95  # 主对角线：库存保持率

# 添加站点间耦合（基于净流量相关性）
for i in range(n_stations):
    for j in range(n_stations):
        if i != j:
            # 如果两个站点净流量符号相反，说明可能有流动关系
            flow_i = avg_net_flow[stations[i]]
            flow_j = avg_net_flow[stations[j]]
            if flow_i * flow_j < 0:  # 异号
                A[i, j] = 0.02  # 小的耦合系数

print(f"    转移率矩阵A: {A.shape}")

# 过程噪声协方差Q
print("  - 估计过程噪声协方差Q...")

# 基于历史库存波动估计
inventory_std = {}
for station in stations:
    inv_series = inventory_data[station].values[:10]  # 前10天
    inventory_std[station] = np.std(inv_series)

Q = np.diag([inventory_std[s]**2 for s in stations])
print(f"    过程噪声协方差Q: {Q.shape}")

# 观测噪声协方差R
print("  - 估计观测噪声协方差R...")

# 假设观测噪声较小（库存数据比较准确）
R = np.eye(n_stations) * 0.5
print(f"    观测噪声协方差R: {R.shape}")

# 观测矩阵H（直接观测库存）
H = np.eye(n_stations)

# ============================================================================
# 3. 卡尔曼滤波预测
# ============================================================================

print("\n[步骤3] 卡尔曼滤波预测...")

def kalman_filter_predict(A, H, Q, R, observations, n_predict=4):
    """
    卡尔曼滤波预测

    参数:
        A: 状态转移矩阵
        H: 观测矩阵
        Q: 过程噪声协方差
        R: 观测噪声协方差
        observations: 历史观测数据 (n_days, n_stations)
        n_predict: 预测天数

    返回:
        predictions: 预测结果 (n_predict, n_stations)
    """
    n_stations = A.shape[0]
    n_days = len(observations)

    # 初始化
    x_hat = observations[0]  # 初始状态估计
    P = Q.copy()  # 初始协方差

    # 滤波阶段（使用历史数据更新状态）
    for t in range(1, n_days):
        # 预测步骤
        x_hat_minus = A @ x_hat
        P_minus = A @ P @ A.T + Q

        # 更新步骤
        y = observations[t]  # 观测值
        K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + R)  # 卡尔曼增益
        x_hat = x_hat_minus + K @ (y - H @ x_hat_minus)
        P = (np.eye(n_stations) - K @ H) @ P_minus

    # 预测阶段（预测未来n_predict天）
    predictions = []
    x_pred = x_hat.copy()
    P_pred = P.copy()

    for t in range(n_predict):
        # 状态预测
        x_pred = A @ x_pred
        P_pred = A @ P_pred @ A.T + Q

        predictions.append(x_pred.copy())

    return np.array(predictions)

# 准备历史观测数据（前10天）
observations = inventory_data[stations].values[:10]

print(f"  - 历史观测数据: {observations.shape}")

# 执行卡尔曼滤波预测
predictions = kalman_filter_predict(A, H, Q, R, observations, n_predict=4)

print(f"  - 预测结果: {predictions.shape}")

# ============================================================================
# 4. 结果整理
# ============================================================================

print("\n[步骤4] 整理预测结果...")

# 创建预测结果DataFrame
dates = pd.date_range('2023-06-11', periods=4)
predictions_df = pd.DataFrame(predictions, columns=stations)
predictions_df.insert(0, '日期', dates)

print("\n  卡尔曼滤波预测结果:")
print(predictions_df.round(2))

# ============================================================================
# 5. 与期望库存对比
# ============================================================================

print("\n[步骤5] 与期望库存对比...")

# 获取实际期望库存（测试集）
actual_target = inventory_data[stations].values[10:14]

# 计算误差
errors = predictions - actual_target
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

print(f"\n  卡尔曼滤波性能:")
print(f"    MAE:  {mae:.2f}")
print(f"    RMSE: {rmse:.2f}")

# 各站点误差
station_errors = pd.DataFrame({
    '站点': stations,
    '平均绝对误差': np.mean(np.abs(errors), axis=0).round(2)
})

print("\n  各站点预测误差:")
print(station_errors)

# ============================================================================
# 6. 保存结果
# ============================================================================

print("\n[步骤6] 保存结果...")

# 保存预测结果
predictions_df.to_csv('问题3_卡尔曼滤波预测结果.csv', index=False, encoding='utf-8-sig')

# 保存模型参数
import pickle

with open('问题3_状态空间模型.pkl', 'wb') as f:
    pickle.dump({
        'A': A,
        'H': H,
        'Q': Q,
        'R': R,
        'stations': stations,
        'performance': {
            'mae': mae,
            'rmse': rmse
        }
    }, f)

print("\n[完成] 结果已保存:")
print("  - 问题3_卡尔曼滤波预测结果.csv")
print("  - 问题3_状态空间模型.pkl")

print("\n" + "="*80)
print("状态空间模型（卡尔曼滤波）完成！")
print("="*80)
