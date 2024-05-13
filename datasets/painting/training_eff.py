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
    plt.figure(figsize=(10, 7))
    for x_values, y_values, label, color, marker in zip(
        x_values_list, y_values_list, labels, colors, markers
    ):
        plt.plot(
            x_values, y_values, label=label, color=color, linewidth=2, marker=marker
        )

    ax=plt.gca()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # x_major_locator=MultipleLocator(4)
    # y_major_locator=MultipleLocator(2)
    # ax.xaxis.set_major_locator(x_major_locator)
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
    plt.savefig("test_rate.svg", dpi=600, format="svg")
    plt.show()

# 示例数据
x_values_list = [
    [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
    [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
    [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
]
y_values_list = [
    [77.4, 81.5, 85.3, 85.6, 85.6, 88.6, 89.3, 89.5, 89.3, 89.4],
    [74.7, 83.4, 84.6, 85.4, 88.9, 89.5, 89.3, 90.2, 89.7, 89.8],
    [85.2, 87.1, 90.3, 91.0, 90.7, 91.3, 91.5, 91.0],
    [81.8, 81.9, 83.4, 82.3, 84.7, 85.8, 84.8, 85.1],
]
labels = ["AIM-8x4", "AIM-16*4", "TimeSformer-8x4", "TimeSformer-16x4"]
colors = ["darkblue", "firebrick", "goldenrod", "yellowgreen"]
markers = ["o", "s", "v", "D"]

# 调用函数绘制多条带有类别的折线图
plot_line_chart(
    x_values_list, y_values_list, "Epoch", "Image-level Acc (%)", "AIM", labels, colors, markers
)
