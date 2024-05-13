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

    ax = plt.gca()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # x_major_locator=MultipleLocator(4)
    # y_major_locator=MultipleLocator(2)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

    major_ticks_top = np.linspace(70, 95, 6)[:-1]
    minor_ticks_top = np.linspace(70, 95, 26)[:-1]
    plt.yticks(major_ticks_top)
    plt.yticks(minor_ticks_top, minor=True)

    ax.grid(linestyle="--", alpha=0.5)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # plt.title(title)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig("test_rate.pdf", dpi=600, format="pdf")
    plt.show()


# 示例数据
x_values_list = [
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
    [1, 2, 3, 4, 5, 6, 7, 8, "x"],
]
y_values_list = [
    [88.1, 88.5, 88.9, 89.6, 89.3, 89.9, 89.9, 89.7, 90.6],
    [88.5, 88.9, 89.4, 89.8, 89.9, 90.5, 90.5, 90.2, 90.9],
    [89.3, 89.8, 90.3, 91.0, 90.7, 91.3, 91.5, 91.0, 92.2],
    [81.8, 81.9, 83.4, 82.3, 84.7, 85.8, 84.8, 85.1, 84.8],
    [85.7, 86.3, 87.4, 87.7, 88.1, 88.9, 88.8, 88.6, 90.5],
    [71.6, 71.9, 73.5, 72.2, 75.3, 77.1, 75.9, 76.3, 76.3],
    [75.1, 75.9, 77.3, 77.5, 79.1, 80.1, 79.8, 79.8, 81.6],
]
labels = [
    "Image-level Acc",
    "Video-level Acc",
    "Video-level Acc (+)",
    "Phase-level F1",
    "Phase-level F1 (+)",
    "Phase-level Jaccard",
    "Phase-level Jaccard (+)",
]
colors = [
    "darkblue",
    "firebrick",
    "goldenrod",
    "yellowgreen",
    "deepskyblue",
    "lightpink",
    "gray",
]
markers = ["o", "s", "v", "D", "*", "P", "X"]

# 调用函数绘制多条带有类别的折线图
plot_line_chart(
    x_values_list,
    y_values_list,
    "Frame Rate $R$",
    "Value",
    "AIM",
    labels,
    colors,
    markers,
)
