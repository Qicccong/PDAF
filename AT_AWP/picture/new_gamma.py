import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import matplotlib

# 清除缓存和恢复默认设置
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
TICK_LABEL_FONT_SIZE = 15  # 刻度标签字体大小 - 确保这是20

# 只保留前三个数据集
datasets = [
    {
        "name": "LastFM",
        "gamma": [0.001, 0.002, 0.005, 0.008, 0.012, 0.016],
        "HR@10": [0.8966, 0.8955, 0.8926, 0.8966, 0.8978, 0.8891],
        "Robust HR@10": [0.8679, 0.8662, 0.8690, 0.8788, 0.8650, 0.8616],
        "y_lim": [0.8, 0.92],  # 统一的y轴范围
        "tick_spacing": 0.04   # 统一的刻度间距
    },
    {
        "name": "AMusic",
        "gamma": [0.001, 0.002, 0.005, 0.008, 0.012, 0.016],
        "HR@10": [0.4420, 0.4347, 0.4398, 0.4403, 0.4369, 0.4206],
        "Robust HR@10": [0.4099, 0.4048, 0.4195, 0.4164, 0.4133, 0.3739],
        "y_lim": [0.3, 0.48],
        "tick_spacing": 0.06
    },
    {
        "name": "ML-1M",
        "gamma": [0.001, 0.002, 0.005, 0.008, 0.012, 0.016],
        "HR@10": [0.7164, 0.7205, 0.7162, 0.7184, 0.7187, 0.7161],
        "Robust HR@10": [0.6427, 0.6281, 0.6409, 0.6384, 0.6288, 0.6293],
        "y_lim": [0.50, 0.75],
        "tick_spacing": 0.08
    },
]

# 可视化配置 - 调整布局为1×3
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
        "figsize": (12, 4),   # 更宽的画布以适应1×3布局
        "wspace": 0.18,      # 增加水平间距
        "hspace": 0.10       # 垂直间距调整为较小值
    }
}

# 创建画布
plt.style.use('default')  # 使用默认样式
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
    x = np.arange(len(dataset["gamma"]))
    
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
    
    # 设置纵轴刻度
    ax.set_ylim(dataset["y_lim"])
    step = dataset["tick_spacing"]
    min_val = dataset["y_lim"][0]
    max_val = dataset["y_lim"][1]
    ticks = np.arange(min_val, max_val + step/2, step)
    ax.set_yticks(ticks)
    
    # === 关键修改1：统一设置刻度标签大小 ===
    # 同时设置x和y轴刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONT_SIZE)
    
    # 设置横轴刻度
    gamma_labels = []
    for gamma in dataset["gamma"]:
        s = f"{gamma:.3f}"
        if s.startswith('0.'):
            s = '0.' + s[2:].rstrip('0').rstrip('.')
        gamma_labels.append(s)
    
    ax.set_xticks(x)
    ax.set_xticklabels(
        gamma_labels,
        # === 关键修改2：移除fontsize参数避免冲突 ===
        # fontsize=TICK_LABEL_FONT_SIZE,  # 注释掉此行
        ha='center',
        fontproperties=font_prop
    )

    # 设置刻度字体 - 注意：我们已经统一设置过大小
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_size(TICK_LABEL_FONT_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_size(TICK_LABEL_FONT_SIZE)
    
    # 网格和字体设置
    ax.grid(linestyle=':', alpha=0.6, axis='y')
    
    # 添加标题
    ax.set_title(dataset["name"], pad=10, fontproperties=font_prop, 
                 fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # 为第一个图添加y轴标签
    # if i == 0:
    #     ax.set_ylabel('HR@10', fontsize=AXIS_LABEL_FONT_SIZE, 
    #                   fontproperties=font_prop, fontweight='bold')
    
    # 为所有图添加x轴标签
    ax.set_xlabel('Gamma', fontsize=AXIS_LABEL_FONT_SIZE, 
                  fontproperties=font_prop, fontweight='bold')

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
plt.savefig('gamma_1x3.png', dpi=300, bbox_inches='tight')
plt.savefig('gamma_1x3.pdf', dpi=300, bbox_inches='tight')
plt.show()