"""
批量更新所有可视化图表的配色方案
使用统一的专业配色，确保所有图表风格一致
"""

import os
import sys

# 导入统一配色方案
from unified_color_scheme import UnifiedColorScheme

# 应用统一配色
UnifiedColorScheme.setup_matplotlib_style()

print("="*80)
print("批量更新可视化图表配色")
print("="*80)

# 需要重新生成的可视化脚本列表
visualization_scripts = [
    ('problem1_visualization.py', '问题一可视化'),
    ('problem2_visualization.py', '问题二可视化'),
    ('problem3_visualization.py', '问题三可视化'),
]

print("\n将重新生成以下可视化图表:")
for script, desc in visualization_scripts:
    if os.path.exists(script):
        print(f"  ✓ {desc} ({script})")
    else:
        print(f"  ✗ {desc} ({script}) - 文件不存在")

print("\n" + "="*80)
print("提示：")
print("  1. 所有图表将使用统一的专业配色方案")
print("  2. 主色调：深蓝色系（#2E5090）")
print("  3. 辅助色：橙色系（#E67E22）、绿色系（#27AE60）")
print("  4. 配色和谐、对比度适中、色盲友好")
print("="*80)

print("\n请分别运行以下命令重新生成图表:")
print("  python problem1_visualization.py")
print("  python problem2_visualization.py")
print("  python problem3_visualization.py")
