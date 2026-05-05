"""问题二 调度时序转换 — 将MILP静态解转换为时间序列调度"""
import pandas as pd, numpy as np

station_info = pd.read_csv('附件数据/站点基础信息.csv', encoding='utf-8')
distance_matrix = pd.read_csv('附件数据/站点距离.csv', index_col=0, encoding='utf-8')
schedule_df = pd.read_csv('问题2_调度方案.csv', encoding='utf-8-sig')

stations = station_info['station_id'].tolist()
v = 20; T_MAX = 300

def travel_time(i, j, num):
    return (distance_matrix.loc[i, j] / v) * 60 + num * 2

# 分配任务到卡车，按距离排序优化先后
tasks = []
for _, row in schedule_df.iterrows():
    tasks.append({
        'truck': row['卡车编号'], 'src': row['起点站点'], 'dst': row['终点站点'],
        'n': int(row['调度量']), 'time': travel_time(row['起点站点'], row['终点站点'], int(row['调度量']))
    })

# 按卡车分拣并计算累计时间
from collections import defaultdict
truck_seq = defaultdict(list)

for t in ['T1','T2','T3']:
    my = [task for task in tasks if task['truck'] == t]
    my.sort(key=lambda x: x['time'])  # 短的先做
    clock = 0; seq = []
    for task in my:
        clock += task['time']
        seq.append({**task, 'start': clock - task['time'], 'end': clock})
    truck_seq[t] = seq
    print(f'{t}: {len(seq)}趟, 累计{clock:.1f}分钟')

# 按时间线展开完整序列
all_events = []
for t in ['T1','T2','T3']:
    for e in truck_seq[t]:
        all_events.append((e['start'], e['end'], e['truck'], e['src'], e['dst'], e['n']))
all_events.sort()

print(f'\n调度时序（按开始时间排序）：')
print(f'{"开始":>6} {"结束":>6} {"卡车":>4} {"起点":>6} {"终点":>6} {"调度量":>6}')
for st, ed, tk, src, dst, n in all_events:
    print(f'{st:6.1f} {ed:6.1f} {tk:>4} {src:>6} {dst:>6} {n:>6}')

print(f'\n时间线占用: {max(e[1] for e in all_events):.1f} / {T_MAX} 分钟')
print(f'时间利用率: {max(e[1] for e in all_events)/T_MAX*100:.1f}%')
