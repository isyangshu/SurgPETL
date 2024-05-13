import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import matplotlib.patches as mpatches

path = "/Users/yangshu/Documents/SurgVideoMAE/Cholec80"
txt_name = [str(i) + ".txt" for i in range(2)]

color_dict = {
    "0": (255, 0, 0),
    "1": (152, 251, 152),
    "2": (135, 206, 250),
    "3": (255, 225, 53),
    "4": (230, 230, 250),
    "5": (237, 145, 33),
    "6": (0, 139, 139),
}

# 加载视频
path_str = str(path)
preds = []
targets = []
inds = []
inds_id = []
for file_name in txt_name:
    if os.path.isfile(os.path.join(path_str, file_name)):
        lines = open(os.path.join(path_str, file_name), "r").readlines()
        for line in lines[1:]:
            line = line.strip()
            name = line.split("[")[0].strip()
            if name in inds:
                continue
            name_id = int(line.split("[")[0].strip().split()[0])

            label = int(line.split("]")[1].split(" ")[1])
            data = np.fromstring(
                line.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            )
            data = softmax(data)
            inds_id.append(name_id)
            inds.append(name)
            preds.append(data)
            targets.append(label)

idxs = np.argsort(np.array(inds_id))
assert len(inds) == len(inds_id)
inds = np.array(inds)[idxs]
preds = np.array(preds)[idxs]
targets = np.array(targets)[idxs]
predicts = np.argmax(preds, axis=1)

# 读取视频的类别序列
vid = np.array([int(ind.split(" ")[1][5:]) for ind in inds])
for v in np.unique(vid):
    sub_inds = np.argwhere(vid == v)
    sub_labels = targets[sub_inds]
    sub_preds = predicts[sub_inds]

    # 定义矩形的宽度和高度
    rectangle_height = 110
    rectangle_width = len(sub_labels)

    # 创建一个空白图像作为画布
    canvas_gt = np.zeros((rectangle_height, rectangle_width, 3), dtype=np.uint8)
    canvas_pred = np.zeros((rectangle_height, rectangle_width, 3), dtype=np.uint8)
    canvas_example = np.ones((rectangle_height*3, 200, 3), dtype=np.uint8) * 255.

    # 遍历视频的每一帧
    for i in range(len(sub_labels)):
        # 获取当前帧对应的类别
        category_gt = sub_labels[i]
        category_pred = sub_preds[i]

        # 在画布上绘制矩形，使用不同的颜色表示不同的类别
        color_gt = color_dict[str(category_gt[0])]
        color_pred = color_dict[str(category_pred[0])]

        cv2.rectangle(
            canvas_gt, (i, 0), (i, rectangle_height - 1), color_gt, thickness=cv2.FILLED
        )
        cv2.rectangle(
            canvas_pred,
            (i, 0),
            (i, rectangle_height - 1),
            color_pred,
            thickness=cv2.FILLED,
        )

    # 绘制图像序列
    fig, (ax1, ax2) = plt.subplots(2)

    plt.subplots_adjust(
        top=0.7, bottom=0.4, left=0.2, right=0.95, hspace=0.15, wspace=0.2
    )
    ax1.set_axis_off()
    ax2.set_axis_off()

    # 显示矩形画布
    ax1.imshow(canvas_gt, cmap="viridis")
    ax1.text(
        -0.25,
        0.5,
        "Ground Truth",
        transform=ax1.transAxes,
        va="center",
        fontweight="bold",
    )

    ax2.imshow(canvas_pred, cmap="viridis")
    ax2.text(
        -0.25, 0.5, "Prediction", transform=ax2.transAxes, va="center", fontweight="bold"
    )

    plt.suptitle(
        "Video" + str(v),
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="0.5", alpha=0.3),
        x=0.5,
        y=0.35,
    )

    plt.show()
    break
