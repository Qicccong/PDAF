import matplotlib.pyplot as plt
import numpy as np

# 设置数据
datasets = [
    {
        "name": "lastfm",
        "gamma": [0.001, 0.002, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020],
        # "HR@10": [0.8966, 0.8955, 0.8845, 0.8926, 0.8932, 0.8983, 0.8932, 0.8978, 0.8891, 0.8966],
        "Robust HR@10": [0.8679, 0.8662, 0.8541, 0.8690, 0.8570, 0.8742, 0.8650, 0.8650, 0.8616, 0.8725]
    },
    {
        "name": "AMusic",
        "gamma": [0.001, 0.002, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020],
         # "HR@10": [0.4420, 0.4347, 0.4206, 0.4398, 0.4324, 0.4403, 0.4443, 0.4369, 0.4206, 0.4403],
        "Robust HR@10": [0.4099, 0.4048, 0.3739, 0.4195, 0.4048, 0.4164, 0.4178, 0.4133, 0.3739, 0.4105]
    },
    {
        "name": "ML-1M",
        "gamma": [0.001, 0.002, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020],
        # "HR@10": [0.7164, 0.7205, 0.7207, 0.7162, 0.7179, 0.7184, 0.0000, 0.7187, 0.7161, 0.7174],
        "Robust HR@10": [0.6427, 0.6281, 0.6272, 0.6409, 0.6232, 0.6384, 0.0000, 0.6288, 0.6293, 0.6310]
    },
    # 第四个数据集的位置（保持空数据）
    # {
    #     "name": "New Dataset",
    #     "gamma": [],
    #     "HR@10": [],
    #     "Robust HR@10": []
    # }
]

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# 颜色配置
colors = {
    "HR@10": "#FF6B6B",    # 红色系
    "Robust HR@10": "#4ECDC4"  # 蓝绿色系
}

markers = {
    "HR@10": "o",
    "Robust HR@10": "s"
}

def calculate_ylim(values, buffer_ratio=0.15):
    """动态计算纵轴范围"""
    valid_values = [v for v in values if v > 0]
    if not valid_values:
        return (0, 1)
    
    min_val = min(valid_values)
    max_val = max(valid_values)
    range_val = max_val - min_val
    
    buffer = range_val * buffer_ratio if range_val > 0 else 0.1
    return (
        max(0, min_val - buffer - 0.05),
        min(1, max_val + buffer + 0.05)
    )

def highlight_max_point(ax, x, y_values, color='red', size=100):
    """突出显示最大值点"""
    valid_values = [y for y in y_values if y > 0]
    if not valid_values:
        return
    
    max_value = max(valid_values)
    max_indices = [i for i, y in enumerate(y_values) if y == max_value and y > 0]
    
    # 绘制红色标记
    for idx in max_indices:
        ax.scatter(
            x[idx], max_value,
            s=size, 
            color=color,
            edgecolor='white',
            linewidth=1.5,
            zorder=10  # 确保显示在最上层
        )

for i in range(4):
    ax = axes[i]
    
    if i < len(datasets):
        dataset = datasets[i]
        x = np.arange(len(dataset["gamma"]))
        
        all_values = []
        for metric in ["HR@10", "Robust HR@10"]:
            if metric in dataset:
                y_values = [y for y in dataset[metric] if y > 0]
                all_values.extend(y_values)
        
        y_min, y_max = calculate_ylim(all_values)
        
        # 先绘制所有折线
        for metric in ["HR@10", "Robust HR@10"]:
            if metric in dataset:
                y_clean = [y if y > 0 else np.nan for y in dataset[metric]]
                ax.plot(
                    x, y_clean,
                    marker=markers[metric],
                    markersize=7,
                    linewidth=2,
                    label=metric,
                    color=colors[metric],
                    markeredgecolor='white',
                    markeredgewidth=1.5
                )
                
                # 添加最大值突出显示（统一用红色）
                highlight_max_point(ax, x, dataset[metric])
        
        ax.set_title(dataset["name"], fontsize=12, pad=12)
        ax.set_xlabel('gamma', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g:.3f}" for g in dataset["gamma"]], rotation=30, fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(ncol=1, loc='lower center', 
                     bbox_to_anchor=(0.5, 1.05),
                     fontsize=10, frameon=True)
    else:
        ax.axis('off')

plt.subplots_adjust(wspace=0.3)
plt.tight_layout()
plt.savefig('highlight_max_points.png', dpi=300, bbox_inches='tight')
plt.show()