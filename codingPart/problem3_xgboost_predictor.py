"""
问题三：XGBoost需求预测模型
基于前10天数据预测未来4天各站点的期望库存
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("问题三：XGBoost需求预测模型")
print("="*80)

# ============================================================================
# 1. 数据加载
# ============================================================================

print("\n[步骤1] 加载数据...")

# 读取数据
riding_data = pd.read_csv('../modelingPart/日汇总骑行数据.csv', encoding='utf-8-sig')
inventory_data = pd.read_csv('../modelingPart/每日初始库存.csv', encoding='utf-8-sig')
weather_data = pd.read_csv('../modelingPart/天气数据.csv', encoding='utf-8-sig')
target_inventory = pd.read_csv('../modelingPart/站点期望库存.csv', encoding='utf-8-sig')
station_info = pd.read_csv('../modelingPart/站点基础信息.csv', encoding='utf-8-sig')

print(f"  - 骑行数据: {len(riding_data)} 条")
print(f"  - 库存数据: {len(inventory_data)} 天")
print(f"  - 天气数据: {len(weather_data)} 天")
print(f"  - 站点信息: {len(station_info)} 个站点")

# ============================================================================
# 2. 特征工程
# ============================================================================

print("\n[步骤2] 特征工程...")

def create_features(riding_df, inventory_df, weather_df, target_df, station_df):
    """
    创建XGBoost训练特征
    """
    features_list = []

    # 转换日期格式
    riding_df['日期'] = pd.to_datetime(riding_df['日期'])
    inventory_df['日期'] = pd.to_datetime(inventory_df['日期'])
    weather_df['日期'] = pd.to_datetime(weather_df['日期'])

    # 获取站点列表
    stations = riding_df['站点 ID'].unique()

    # 为每个站点每一天创建特征
    for station_id in stations:
        # 获取该站点的骑行数据
        station_riding = riding_df[riding_df['站点 ID'] == station_id].sort_values('日期').reset_index(drop=True)

        # 获取站点信息
        station_row = station_df[station_df['station_id'] == station_id].iloc[0]
        target_row = target_df[target_df['站点ID'] == station_id].iloc[0]

        # 站点类型编码
        type_mapping = {'居民区': 0, 'residential': 0,
                       '商务区': 1, 'business': 1,
                       '教育区': 2, 'education': 2,
                       '交通枢纽': 3, 'transport': 3}
        station_type_code = type_mapping.get(station_row['类型'], 0)

        for idx, row in station_riding.iterrows():
            date = row['日期']

            # 获取库存数据
            inv_row = inventory_df[inventory_df['日期'] == date]
            if len(inv_row) == 0:
                continue
            current_inventory = inv_row[station_id].values[0]

            # 获取天气数据
            weather_row = weather_df[weather_df['日期'] == date].iloc[0]

            # 基础特征
            features = {
                'station_id': station_id,
                'date': date,

                # 时间特征
                'day_of_week': weather_row['星期'],
                'is_weekend': 1 if weather_row['星期'] in [6, 7] else 0,
                'is_holiday': weather_row['是否节假日'],
                'day_of_month': date.day,

                # 站点特征
                'station_type': station_type_code,
                'station_capacity': station_row['容量'],
                'target_weekday': target_row['工作日期望库存'],
                'target_weekend': target_row['周末期望库存'],

                # 天气特征
                'temperature': weather_row['平均温度'],
                'weather_coef': weather_row['天气系数'],

                # 当前状态
                'current_inventory': current_inventory,
                'total_outflow': row['总流出'],
                'total_inflow': row['总流入'],
                'net_flow': row['净流量'],
                'morning_outflow': row['早高峰流出'],
                'morning_inflow': row['早高峰流入'],
                'evening_outflow': row['晚高峰流出'],
                'evening_inflow': row['晚高峰流入'],
            }

            # 滞后特征（前1天、前7天）
            if idx >= 1:
                prev_row = station_riding.iloc[idx-1]
                prev_inv_row = inventory_df[inventory_df['日期'] == prev_row['日期']]
                if len(prev_inv_row) > 0:
                    features['inventory_lag1'] = prev_inv_row[station_id].values[0]
                    features['net_flow_lag1'] = prev_row['净流量']
                else:
                    features['inventory_lag1'] = current_inventory
                    features['net_flow_lag1'] = 0
            else:
                features['inventory_lag1'] = current_inventory
                features['net_flow_lag1'] = 0

            if idx >= 7:
                prev7_row = station_riding.iloc[idx-7]
                prev7_inv_row = inventory_df[inventory_df['日期'] == prev7_row['日期']]
                if len(prev7_inv_row) > 0:
                    features['inventory_lag7'] = prev7_inv_row[station_id].values[0]
                else:
                    features['inventory_lag7'] = current_inventory
            else:
                features['inventory_lag7'] = current_inventory

            # 滚动统计特征（前3天）
            if idx >= 3:
                recent_3_inv = []
                for i in range(1, 4):
                    prev_date = station_riding.iloc[idx-i]['日期']
                    prev_inv = inventory_df[inventory_df['日期'] == prev_date]
                    if len(prev_inv) > 0:
                        recent_3_inv.append(prev_inv[station_id].values[0])

                if len(recent_3_inv) > 0:
                    features['inventory_rolling_mean_3'] = np.mean(recent_3_inv)
                    features['inventory_rolling_std_3'] = np.std(recent_3_inv)
                else:
                    features['inventory_rolling_mean_3'] = current_inventory
                    features['inventory_rolling_std_3'] = 0
            else:
                features['inventory_rolling_mean_3'] = current_inventory
                features['inventory_rolling_std_3'] = 0

            # 交互特征
            features['inventory_x_weekend'] = current_inventory * features['is_weekend']
            features['type_x_dayofweek'] = station_type_code * features['day_of_week']
            features['netflow_x_weather'] = row['净流量'] * weather_row['天气系数']

            # 目标变量：根据工作日/周末选择期望库存
            if features['is_weekend'] == 1:
                features['target'] = target_row['周末期望库存']
            else:
                features['target'] = target_row['工作日期望库存']

            features_list.append(features)

    return pd.DataFrame(features_list)

# 创建特征数据集
df_features = create_features(riding_data, inventory_data, weather_data,
                              target_inventory, station_info)

print(f"  - 特征数据集: {len(df_features)} 条记录")
print(f"  - 特征维度: {len(df_features.columns)-3} 个特征")  # 减去station_id, date, target

# ============================================================================
# 3. 数据划分
# ============================================================================

print("\n[步骤3] 数据划分...")

# 前10天作为训练集，后4天作为测试集
train_data = df_features[df_features['date'] < '2023-06-11']
test_data = df_features[df_features['date'] >= '2023-06-11']

print(f"  - 训练集: {len(train_data)} 条 (前10天)")
print(f"  - 测试集: {len(test_data)} 条 (后4天)")

# 特征列（排除station_id, date, target）
feature_cols = [col for col in df_features.columns
                if col not in ['station_id', 'date', 'target']]

X_train = train_data[feature_cols]
y_train = train_data['target']
X_test = test_data[feature_cols]
y_test = test_data['target']

print(f"  - 训练特征维度: {X_train.shape}")
print(f"  - 测试特征维度: {X_test.shape}")

# ============================================================================
# 4. 模型训练
# ============================================================================

print("\n[步骤4] 训练XGBoost模型...")

# XGBoost参数
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,  # L1正则化
    'reg_lambda': 1.0,  # L2正则化
    'random_state': 42,
    'n_jobs': -1
}

# 训练模型
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train,
         eval_set=[(X_train, y_train), (X_test, y_test)],
         verbose=False)

print("  [OK] 模型训练完成")

# ============================================================================
# 5. 模型评估
# ============================================================================

print("\n[步骤5] 模型评估...")

# 训练集预测
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print(f"\n  训练集性能:")
print(f"    MAE:  {train_mae:.2f}")
print(f"    RMSE: {train_rmse:.2f}")
print(f"    R2:   {train_r2:.4f}")

# 测试集预测
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f"\n  测试集性能:")
print(f"    MAE:  {test_mae:.2f}")
print(f"    RMSE: {test_rmse:.2f}")
print(f"    MAPE: {test_mape:.2f}%")
print(f"    R2:   {test_r2:.4f}")

# ============================================================================
# 6. 特征重要性
# ============================================================================

print("\n[步骤6] 特征重要性分析...")

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 10 重要特征:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# 7. 生成预测结果
# ============================================================================

print("\n[步骤7] 生成预测结果...")

# 为测试集添加预测值
test_results = test_data[['station_id', 'date', 'target']].copy()
test_results['predicted'] = y_test_pred
test_results['error'] = test_results['predicted'] - test_results['target']
test_results['abs_error'] = np.abs(test_results['error'])

# 按站点汇总
station_summary = test_results.groupby('station_id').agg({
    'target': 'mean',
    'predicted': 'mean',
    'abs_error': 'mean'
}).round(2)
station_summary.columns = ['平均期望库存', '平均预测库存', '平均绝对误差']

print("\n  各站点预测性能:")
print(station_summary)

# 保存预测结果
test_results.to_csv('问题3_XGBoost预测结果.csv', index=False, encoding='utf-8-sig')
feature_importance.to_csv('问题3_特征重要性.csv', index=False, encoding='utf-8-sig')

print("\n[完成] 预测结果已保存:")
print("  - 问题3_XGBoost预测结果.csv")
print("  - 问题3_特征重要性.csv")

# ============================================================================
# 8. 保存模型
# ============================================================================

import pickle

with open('问题3_XGBoost模型.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'feature_cols': feature_cols,
        'params': params,
        'performance': {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'test_r2': test_r2
        }
    }, f)

print("  - 问题3_XGBoost模型.pkl")

print("\n" + "="*80)
print("XGBoost预测模型训练完成！")
print("="*80)
