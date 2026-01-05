import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_error_analysis():
    # === 1. 真实数据录入 ===
    # 格式: '真实->误判'
    error_types = ['E → S', 'N → M', 'M → N', 'U → R', 'B → V', 'X → S']

    # 对应的错误数量
    error_counts = [30, 16, 7, 7, 5, 5]

    # 定义颜色：最严重的用深红，其他的用橙色
    colors = ['#D32F2F', '#F57C00', '#FF9800', '#FFB74D', '#FFCC80', '#FFE0B2']

    # === 2. 绘图 ===
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 画横向柱状图 (Horizontal Bar) 看起来更像排行榜
    y_pos = np.arange(len(error_types))
    bars = ax.barh(y_pos, error_counts, color=colors, height=0.6)

    # y轴反转，让最大的错误排在最上面
    ax.invert_yaxis()

    # === 3. 美化 ===
    ax.set_xlabel('误判样本数量 (Count)', fontsize=12, fontweight='bold')
    ax.set_title('Top-6 高频混淆错误分布分析', fontsize=14, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(error_types, fontsize=12, fontweight='bold')

    # 去掉上边和右边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加垂直网格线
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # === 4. 标注数值和原因 ===
    # 定义每个错误的简短原因
    reasons = [
        "拇指位置重叠 (最大痛点)",
        "指间覆盖相似",
        "指间覆盖相似",
        "手指交叉深度丢失",
        "手指开合特征模糊",
        "食指弯曲度近似握拳"
    ]

    for i, bar in enumerate(bars):
        width = bar.get_width()
        # 在柱子末尾标数字
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}',
                va='center', fontweight='bold', fontsize=12, color='black')

        # 在柱子内部或旁边标原因
        ax.text(1, bar.get_y() + bar.get_height() / 2,
                reasons[i],
                va='center', ha='left', fontsize=10, color='white' if width > 10 else 'black', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/top_errors_analysis.png')
    print("真实错误分析图已生成: results/top_errors_analysis.png")
    plt.show()


if __name__ == "__main__":
    plot_error_analysis()