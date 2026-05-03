"""
统一配色方案 - 为所有可视化图表提供一致的专业配色
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 统一配色方案定义
# ============================================================================

class UnifiedColorScheme:
    """
    统一配色方案
    基于专业的数据可视化配色理论，确保：
    1. 色彩和谐
    2. 对比度适中
    3. 色盲友好
    4. 适合打印和屏幕显示
    """

    # 主色调：深蓝色系（专业、稳重）
    PRIMARY = '#2E5090'      # 深蓝
    PRIMARY_LIGHT = '#5B7FC7'  # 浅蓝
    PRIMARY_DARK = '#1A3A6B'   # 深蓝

    # 辅助色调：橙色系（活力、对比）
    SECONDARY = '#E67E22'    # 橙色
    SECONDARY_LIGHT = '#F39C12'  # 浅橙
    SECONDARY_DARK = '#D35400'   # 深橙

    # 强调色：绿色系（成功、增长）
    ACCENT = '#27AE60'       # 绿色
    ACCENT_LIGHT = '#2ECC71'   # 浅绿
    ACCENT_DARK = '#229954'    # 深绿

    # 警告色：红色系（警告、减少）
    WARNING = '#E74C3C'      # 红色
    WARNING_LIGHT = '#EC7063'  # 浅红
    WARNING_DARK = '#C0392B'   # 深红

    # 中性色：灰色系（背景、文字）
    NEUTRAL_DARK = '#2C3E50'   # 深灰（文字）
    NEUTRAL = '#7F8C8D'        # 中灰
    NEUTRAL_LIGHT = '#BDC3C7'  # 浅灰
    NEUTRAL_LIGHTER = '#ECF0F1'  # 极浅灰（背景）

    # 分类色板（用于多类别对比）
    CATEGORICAL = [
        '#2E5090',  # 深蓝
        '#E67E22',  # 橙色
        '#27AE60',  # 绿色
        '#E74C3C',  # 红色
        '#9B59B6',  # 紫色
        '#16A085',  # 青色
        '#F39C12',  # 黄色
        '#34495E',  # 深灰蓝
        '#E91E63',  # 粉红
        '#00BCD4',  # 浅蓝
    ]

    # 渐变色板（用于热力图、连续数据）
    SEQUENTIAL_BLUE = ['#EBF5FB', '#AED6F1', '#5DADE2', '#2E86C1', '#1B4F72']
    SEQUENTIAL_ORANGE = ['#FEF5E7', '#FAD7A0', '#F8C471', '#E67E22', '#BA4A00']
    SEQUENTIAL_GREEN = ['#E8F8F5', '#A9DFBF', '#52BE80', '#27AE60', '#186A3B']

    # 发散色板（用于正负对比）
    DIVERGING = ['#E74C3C', '#EC7063', '#F8F9F9', '#5DADE2', '#2E5090']

    @classmethod
    def get_palette(cls, n_colors=None, palette_type='categorical'):
        """
        获取调色板

        参数:
            n_colors: 需要的颜色数量
            palette_type: 'categorical', 'sequential_blue', 'sequential_orange',
                         'sequential_green', 'diverging'
        """
        if palette_type == 'categorical':
            colors = cls.CATEGORICAL
        elif palette_type == 'sequential_blue':
            colors = cls.SEQUENTIAL_BLUE
        elif palette_type == 'sequential_orange':
            colors = cls.SEQUENTIAL_ORANGE
        elif palette_type == 'sequential_green':
            colors = cls.SEQUENTIAL_GREEN
        elif palette_type == 'diverging':
            colors = cls.DIVERGING
        else:
            colors = cls.CATEGORICAL

        if n_colors is None:
            return colors
        elif n_colors <= len(colors):
            return colors[:n_colors]
        else:
            # 如果需要更多颜色，循环使用
            return (colors * ((n_colors // len(colors)) + 1))[:n_colors]

    @classmethod
    def setup_matplotlib_style(cls):
        """
        设置matplotlib全局样式
        """
        plt.rcParams.update({
            # 字体设置 - Microsoft YaHei优先
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
            'axes.unicode_minus': False,

            # 图形质量
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',

            # 颜色设置
            'axes.prop_cycle': plt.cycler(color=cls.CATEGORICAL),
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',

            # 网格设置
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.linewidth': 0.8,
            'grid.color': cls.NEUTRAL_LIGHT,

            # 轴设置
            'axes.edgecolor': cls.NEUTRAL_DARK,
            'axes.linewidth': 1.2,
            'axes.labelcolor': cls.NEUTRAL_DARK,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.titlecolor': cls.NEUTRAL_DARK,

            # 刻度设置
            'xtick.color': cls.NEUTRAL_DARK,
            'ytick.color': cls.NEUTRAL_DARK,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,

            # 图例设置
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.facecolor': 'white',
            'legend.edgecolor': cls.NEUTRAL_LIGHT,
            'legend.fontsize': 10,

            # 线条设置
            'lines.linewidth': 2,
            'lines.markersize': 8,

            # 其他
            'patch.edgecolor': cls.NEUTRAL_DARK,
            'patch.linewidth': 1,
        })

        # 设置seaborn样式
        sns.set_palette(cls.CATEGORICAL)
        sns.set_style("whitegrid", {
            'axes.edgecolor': cls.NEUTRAL_DARK,
            'grid.color': cls.NEUTRAL_LIGHT,
        })

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 应用统一配色方案
    UnifiedColorScheme.setup_matplotlib_style()

    print("="*80)
    print("统一配色方案")
    print("="*80)

    print("\n主色调:")
    print(f"  PRIMARY: {UnifiedColorScheme.PRIMARY}")
    print(f"  SECONDARY: {UnifiedColorScheme.SECONDARY}")
    print(f"  ACCENT: {UnifiedColorScheme.ACCENT}")
    print(f"  WARNING: {UnifiedColorScheme.WARNING}")

    print("\n分类色板（10色）:")
    for i, color in enumerate(UnifiedColorScheme.CATEGORICAL, 1):
        print(f"  {i}. {color}")

    print("\n配色方案已应用到matplotlib全局设置")
    print("="*80)
