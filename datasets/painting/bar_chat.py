import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data
data = {
    'Group': ['TCGA_BRCA', 'TCGA_COADREAD', 'TGCA_KIRP', 'TCGA_KIRC', 'TCGA_STAD', 'TCGA_LUAD', 'TCGA_UCEC'],
    'DinoV2_pre': [68.535, 66.96, 77.85, 70.4025, 61.755, 61.62, 70.7175],
    'DinoV2': [64.635, 65.1225, 73.2375, 71.8125, 60.3025, 62.4725, 67.0075],
    'plip': [60, 63.345, 73.8925, 65.7475, 59.58, 59.3425, 69.7325],
    'ResNet-50': [60.385, 59.87, 66.8875, 65.89, 60.738, 60.545, 65.153],
}

df = pd.DataFrame(data)

colors = ['#FF6347', '#4682B4', '#32CD32', '#DA70D6']
colors = colors[::-1]
DinoV2_pre_mean = df['DinoV2_pre'].mean()
DinoV2_mean = df['DinoV2'].mean()
plip_mean = df['plip'].mean()
resnet_mean = df['ResNet-50'].mean()

# Number of groups
n_groups = len(df)

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(20, 5))

# Set the positions and width for the bars
index = np.arange(n_groups)
bar_width = 0.18

# Plot the bars with error bars
rects1 = ax.bar(index, df['DinoV2_pre'], bar_width, color=colors[0], label='DinoV2++ (ours)', alpha=0.25)
rects2 = ax.bar(index + bar_width, df['DinoV2'], bar_width, color=colors[1], label='DinoV2', alpha=0.25)
rects3 = ax.bar(index + 2 * bar_width, df['plip'], bar_width, color=colors[2], label='plip', alpha=0.25)
rects4 = ax.bar(index + 3 * bar_width, df['ResNet-50'], bar_width, color=colors[3], label='ResNet-50', alpha=0.25)

plt.axhline(DinoV2_pre_mean, color=colors[0], linewidth=2, linestyle='dashed', alpha=0.65)
plt.axhline(DinoV2_mean, color=colors[1], linewidth=2, linestyle='dashed', alpha=0.65)
plt.axhline(plip_mean, color=colors[2], linewidth=2, linestyle='dashed', alpha=0.65)
plt.axhline(resnet_mean, color=colors[3], linewidth=2, linestyle='dashed', alpha=0.65)



# Add labels, title, and axes ticks
# ax.set_xlabel('Group')
ax.set_ylabel('C-index(%)')
ax.set_title('Results on different Feature Extractor')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(df['Group'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(ncol=4)
# Show the figure
plt.show()