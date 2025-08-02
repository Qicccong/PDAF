import matplotlib.pyplot as plt
import numpy as np

# 设置数据
datasets = [
    {
        "name": "lastfm",
        "learning_rates": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        # "HR@10": [0.8874, 0.8891, 0.8983, 0.8822, 0.8851, 0.8949],
        "Robust HR@10":[0.8154, 0.8397, 0.8742, 0.8581, 0.8553, 0.8742]
    },
    {
        "name": "AMusic",
        "learning_rates": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        # "HR@10": [0.4206, 0.4206, 0.4217, 0.4403, 0.4206, 0.4223],
        "Robust HR@10":[0.3739, 0.3739, 0.3840, 0.4164, 0.3739, 0.3992],
    },
    {
        "name": "ML-1M",
        "learning_rates": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
        # "HR@10": [0.7195,0.7189,0.7164,0.7209,0.7197, 0.0000],
        "Robust HR@10":[0.5397,0.6113,0.6427,0.6336,0.6051, 0.0000],
    },
    # {
    #     "name": "Yelp",
    #     "learning_rates": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    #     # "HR@10": [0.7195,0.7189,0.7164,0.7209,0.7197, 0.0000],
    #     "Robust HR@10":[0.5397,0.6113,0.6427,0.6336,0.6051, 0.0000],
    # },
]

for data in datasets:
    for key, value in data.items():
        print(key, value)
        if key == "Robust HR@10":
            new_v = [v * 100 for v in value]
            data[key] = new_v

print(datasets)

# 设置图形
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 设置颜色
colors = {
    "HR@10": "#2C91E0",
    "Robust HR@10": "#2C91E0"
}

# 绘制每个数据集的堆叠柱状图
for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    # 设置x轴位置
    x = np.arange(len(dataset["learning_rates"]))
    
    # 绘制堆叠柱状图
    ax.bar(x, dataset["Robust HR@10"], width=0.6, label='Robust HR@10', color=colors["Robust HR@10"])
    
    # 设置标题和标签
    ax.set_title(dataset["name"], fontsize=12, fontfamily='sans-serif')
    ax.set_xlabel('Learning Rate', fontsize=10, fontfamily='sans-serif')
    # ax.set_ylabel('HR@10', fontsize=10, fontfamily='sans-serif')
    
    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels([str(lr) for lr in dataset["learning_rates"]], rotation=0, fontsize=9)
    
    # 添加图例
    # ax.legend(loc='upper right', fontsize=9)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

# 手动调整子图间距
plt.subplots_adjust(wspace=0.05)  # 设置柱状图之间的宽度间距

# 调整布局
plt.tight_layout()

# 保存图形
plt.show()