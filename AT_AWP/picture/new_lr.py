import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import matplotlib

# 强制清除所有Matplotlib缓存和默认设置
plt.close('all')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# 设置字体
font_path = 'AT_AWP/picture/Times New Roman.ttf'
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = font_prop.get_name()
matplotlib.rcParams['mathtext.it'] = font_prop.get_name() + ':italic'
matplotlib.rcParams['mathtext.bf'] = font_prop.get_name() + ':bold'

# 直接定义字体大小常量 - 直接控制
LEGEND_FONT_SIZE = 16      # 图例字体大小
TITLE_FONT_SIZE = 16       # 子图标题字体大小
AXIS_LABEL_FONT_SIZE = 18  # 坐标轴标签字体大小
TICK_LABEL_FONT_SIZE = 16  # 刻度标签字体大小
VALUE_FONT_SIZE = 16       # 数值标签字体大小（柱子上的数字）

# 只保留前三个数据集
datasets = [
    {
        "name": "LastFM",
        "learning_rates": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        "HR@10": [0.8874, 0.8891, 0.8966, 0.8822, 0.8851, 0.8949],
        "Robust HR@10":[0.8154, 0.8397, 0.8788, 0.8581, 0.8553, 0.8742],
        "y_lim": [0.75, 0.92],
        "tick_spacing": 0.06
    },
    {
        "name": "AMusic",
        "learning_rates": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        "HR@10": [0.4206, 0.4206, 0.4217, 0.4403, 0.4206, 0.4223],
        "Robust HR@10":[0.3739, 0.3739, 0.3840, 0.4164, 0.3739, 0.3992],
        "y_lim": [0.3, 0.46],
        "tick_spacing": 0.05
    },
    {
        "name": "ML-1M",
        "learning_rates": [0.005, 0.001, 0.002, 0.005, 0.01, 0.02],
        "HR@10": [0.7195, 0.7195, 0.7189, 0.7163, 0.7209, 0.7197],
        "Robust HR@10":[0.5397, 0.5397, 0.6113, 0.6427, 0.6336, 0.6051],
        "y_lim": [0.50, 0.75],
        "tick_spacing": 0.08
    }
]

# 修改为1×3布局的可视化配置
config = {
    "colors": {
        "HR@10": "#DF6D3B",  # 橙色
        "Robust HR@10": "#2787C7"  # 蓝色
    },
    "style": {
        "bar": {
            "width": 0.6, 
            "edgecolor": "black", 
            "linewidth": 1
        }
    },
    "layout": {
        "figsize": (12, 4),  # 更宽的画布以适应1×3布局
        "wspace": 0.18,      # 增加水平间距
        "hspace": 0.10       # 垂直间距调整为较小值
    }
}

# 创建画布
plt.style.use('default')  # 使用最简单的默认样式
fig = plt.figure(figsize=config["layout"]["figsize"], dpi=120)
# 创建1×3的网格布局
gs = GridSpec(1, 3, figure=fig, 
              wspace=config["layout"]["wspace"], 
              hspace=config["layout"]["hspace"])

# 准备图例元素
handles = [
    plt.Rectangle((0,0),1,1, color=config["colors"]["Robust HR@10"], label="Robust HR@10"),
    plt.Rectangle((0,0),1,1, color=config["colors"]["HR@10"], label="HR@10")
]

# 为每个数据集创建子图
for i, dataset in enumerate(datasets):
    ax = fig.add_subplot(gs[0, i])  # 所有图都在第一行
    x = np.arange(len(dataset["learning_rates"]))
    
    # 堆叠柱状图
    bars_bottom = ax.bar(
        x, 
        dataset["Robust HR@10"],
        color=config["colors"]["Robust HR@10"],
        **config["style"]["bar"]
    )
    
    delta_values = [h - r for h, r in zip(dataset["HR@10"], dataset["Robust HR@10"])]
    bars_top = ax.bar(
        x, 
        delta_values,
        bottom=dataset["Robust HR@10"],
        color=config["colors"]["HR@10"],
        **config["style"]["bar"]
    )
    
    # 设置纵轴刻度 - 直接控制刻度字体大小
    ax.set_ylim(dataset["y_lim"])
    step = dataset["tick_spacing"]
    min_val = dataset["y_lim"][0]
    max_val = dataset["y_lim"][1]
    ticks = np.arange(min_val, max_val + step/2, step)
    ax.set_yticks(ticks)
    
    # === 关键修改：统一设置刻度标签大小 ===
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONT_SIZE)
    
    # 坐标轴设置
    lr_labels = []
    for lr in dataset["learning_rates"]:
        s = f"{lr:.3f}"
        if s.startswith('0.'):
            s = '0.' + s[2:].rstrip('0').rstrip('.')
        lr_labels.append(s)
    
    ax.set_xticks(x)
    ax.set_xticklabels(
        lr_labels,
        ha='center',
        fontproperties=font_prop
    )

    # 强制加粗刻度标签 - 关键修改
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_size(TICK_LABEL_FONT_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_size(TICK_LABEL_FONT_SIZE)
    
    # 网格和字体设置
    ax.grid(linestyle=':', alpha=0.6, axis='y')
    
    # 添加标题 - 直接使用定义的字号
    ax.set_title(dataset["name"], pad=10, fontproperties=font_prop, 
                 fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # 为所有图添加x轴标签
    ax.set_xlabel('Learning rate', fontsize=AXIS_LABEL_FONT_SIZE, 
                  fontproperties=font_prop, fontweight='bold')
    
    # 为第一个图添加y轴标签
    # if i == 0:
    #     ax.set_ylabel('HR@10', fontsize=AXIS_LABEL_FONT_SIZE, 
    #                   fontproperties=font_prop, fontweight='bold')

# 添加图例
legend_font_prop = fm.FontProperties(
    fname=font_path,
    size=LEGEND_FONT_SIZE
)
# 将图例放在中心位置上方
legend = fig.legend(
    handles=handles, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.05), 
    ncol=2, 
    prop=legend_font_prop,
    frameon=True,
    fancybox=True
)

# 调整布局
fig.subplots_adjust(
    left=0.07,    
    right=0.97,   
    bottom=0.2,  # 为图例留出空间
    top=0.90,    
    wspace=config["layout"]["wspace"],
    hspace=config["layout"]["hspace"]
)

# 保存图像
plt.savefig('learning_rate_1x3.png', dpi=300, bbox_inches='tight')
plt.savefig('learning_rate_1x3.pdf', dpi=300, bbox_inches='tight')
plt.show()