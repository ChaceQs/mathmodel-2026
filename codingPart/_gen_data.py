import pandas as pd
f = pd.read_csv('站点特征分析结果.csv', encoding='utf-8')

for label, name in [(0, '供给型(居民区)'), (2, '需求型(商务区)'), (1, '平衡型(交通/教育)')]:
    g = f[f['聚类标签'] == label]
    print(f'=== {name} (n={len(g)}) ===')
    print(f'  流出={g["平均流出"].mean():.2f} 流入={g["平均流入"].mean():.2f} 净流量={g["平均净流量"].mean():.2f}')
    print(f'  潮汐指数={g["潮汐指数"].mean():.2f}')
    print(f'  早高峰净流出={g["早高峰净流出"].mean():.2f} 晚高峰净流入={g["晚高峰净流入"].mean():.2f}')
    print(f'  工作日流出={g["工作日平均流出"].mean():.2f} 周末流出={g["周末平均流出"].mean():.2f}')
    print()

print('=== 各站点流量（用于表格） ===')
for _, r in f.iterrows():
    print(f'{r["站点名称"]:6s}: 流出={r["平均流出"]:.2f} 流入={r["平均流入"]:.2f} 净={r["平均净流量"]:.2f} 潮汐={r["潮汐指数"]:.2f}')
