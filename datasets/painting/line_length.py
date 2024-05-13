import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

def plot_line_chart(
    x_values_list, y_values_list, x_label, y_label, title, labels, colors, markers
):
    """
    绘制多条带有类别的折线图

    参数：
    x_values_list: 包含多条折线的 x 轴数据点列表的列表
    y_values_list: 包含多条折线的 y 轴数据点列表的列表
    x_label: x 轴的标签
    y_label: y 轴的标签
    title: 图表的标题
    labels: 类别标签的列表
    colors: 颜色的列表
    markers: 点形状的列表
    """
    plt.figure(figsize=(12, 8))
    for x_values, y_values, label, color, marker in zip(
        x_values_list, y_values_list, labels, colors, markers
    ):
        plt.plot(
            x_values, y_values, label=label, color=color, linewidth=2, marker=marker
        )

    ax=plt.gca()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    x_major_locator=MultipleLocator(4)
    # y_major_locator=MultipleLocator(2)
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)


    major_ticks_top = np.linspace(70,95,6)[:-1]
    minor_ticks_top = np.linspace(70,95,26)[:-1]
    plt.yticks(major_ticks_top)
    plt.yticks(minor_ticks_top, minor=True)

    ax.grid(linestyle='--', alpha=0.5)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    # plt.title(title)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig("test_length.pdf", dpi=600, format="pdf")
    plt.show()

# 示例数据
x_values_list = [
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 20, 24],
]
y_values_list = [
    [88.1, 89.6, 89.5, 90.2, 90.3, 91.0],
    [88.6, 89.8, 90.1, 90.8, 91.1, 91.6],
    [89.5, 91.0, 90.9, 91.6, 91.9, 92.5],
    [82.1, 82.3, 85.1, 85.7, 86.2, 86.7],
    [86.3, 87.7, 88.1, 89.0, 89.4, 90.1],
    [71.8, 72.2, 76.2, 77.4, 78.2, 78.1],
    [75.8, 77.5, 79.2, 80.6, 81.5, 82.2]
]
labels = ["Image-level Acc", "Video-level Acc", "Video-level Acc (+)", "Phase-level F1", "Phase-level F1 (+)", "Phase-level Jaccard", "Phase-level Jaccard (+)"]
colors = ["darkblue", "firebrick", "goldenrod", "yellowgreen", "deepskyblue", "lightpink", "gray"]
markers = ["o", "s", "v", "D", "*", "P", "X"]

# 调用函数绘制多条带有类别的折线图
plot_line_chart(
    x_values_list, y_values_list, "Sequence Length $T$", "Value", "AIM", labels, colors, markers
)
