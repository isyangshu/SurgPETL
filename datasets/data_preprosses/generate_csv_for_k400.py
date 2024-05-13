import os
import shutil
import csv

# CSV文件路径
csv_file = "/home/syangcw/adapt-image-models/data/kinetics400/annotations/kinetics_val.csv"

# 目标子文件夹路径
destination_csv = "/jhcnas4/syangcw/Kinetics-400/val.csv"
video_folder = "/jhcnas4/syangcw/Kinetics-400/videos_val"
print(len(os.listdir(video_folder)))
categoris = set()
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # 跳过CSV文件的标题行
    for row in csv_reader:
        video_name = row[1]  # 视频文件名
        category = row[0]  # 视频对应的类别
        categoris.add(category)

categoris = sorted(list(categoris))
print(len(categoris))
cat_dict = {}

for i in range(len(categoris)):
    cat_dict[categoris[i]] = i
print(cat_dict)
data_list = []

count_remove = 0
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # 跳过CSV文件的标题行
    for row in csv_reader:
        video_name = row[1]  # 视频文件名
        category = row[0]  # 视频对应的类别
        video_path = os.path.join(video_folder, video_name+'.mp4')
        if os.path.exists(video_path):
            data_list.append([video_path, cat_dict[category]])
        else:
            count_remove += 1
print(count_remove)
with open(destination_csv, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')

    # 循环写入每一行数据
    for row in data_list:
        writer.writerow(row)

print("CSV 文件写入完成。", len(data_list))      