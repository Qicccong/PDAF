import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import matplotlib
import os

# 强制清除所有Matplotlib缓存和默认设置
plt.close('all')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# 确保使用Times New Roman字体的可靠方法
def setup_times_new_roman():
    """强制使用Times New Roman字体，解决常见错误"""
    try:
        # 方法1：直接使用字体名称（如果系统安装了Times New Roman）
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        
        # 验证字体是否存在
        test_font = fm.FontProperties(family='Times New Roman')
        _ = fm.findfont(test_font)
        print("成功通过字体名称加载Times New Roman")
        return test_font
    except ValueError:
        pass
    
    try:
        # 方法2：使用绝对路径加载字体文件
        # 需要将路径替换为您的Times New Roman字体文件的实际位置
        font_path = os.path.abspath('AT_AWP/picture/Times New Roman.ttf')
        
        if not os.path.exists(font_path):
            # 尝试在Windows字体目录中查找
            win_font_path = os.path.join(os.environ['WINDIR'], 'Fonts', 'times.ttf')
            if os.path.exists(win_font_path):
                font_path = win_font_path
            else:
                # 尝试在Linux/Mac常见位置查找
                possible_paths = [
                    '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
                    '/Library/Fonts/Times New Roman.ttf',
                    '/System/Library/Fonts/Times New Roman.ttf'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        font_path = path
                        break
        
        font_prop = fm.FontProperties(fname=font_path)
        
        # 设置全局字体
        matplotlib.rcParams['font.family'] = font_prop.get_name()
        print(f"成功通过文件加载Times New Roman: {font_path}")
        return font_prop
    except Exception as e:
        print(f"无法加载Times New Roman: {e}")
        print("使用备用字体方案")
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times', 'Liberation Serif']
        return fm.FontProperties(family='serif')

# 设置字体
font_prop = setup_times_new_roman()

# 如果上述方法失败，强制设置数学字体为STIX（类似于Times New Roman）
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# ========== 字体大小设置 - 关键修复 ==========
# 直接定义字体大小常量
TITLE_FONT_SIZE = 14       # 标题字体大小
AXIS_LABEL_FONT_SIZE = 22  # 坐标轴标签字体大小
TICK_LABEL_FONT_SIZE = 18  # 刻度标签字体大小
LEGEND_FONT_SIZE = 18      # 图例字体大小
DATA_LABEL_FONT_SIZE = 16   # 数据标签字体大小

# 全局设置字体大小
matplotlib.rcParams['font.size'] = TICK_LABEL_FONT_SIZE
matplotlib.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
matplotlib.rcParams['axes.labelsize'] = AXIS_LABEL_FONT_SIZE
matplotlib.rcParams['xtick.labelsize'] = TICK_LABEL_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_LABEL_FONT_SIZE
matplotlib.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE

# 从表格中提取的数据
models = ["MLP", "MLP+RAT", "MLP+RAWP", "MLP+RAWP-FT", "MLP+RAWP-PDAF"]

# HR@10指标数据
hr_lastfm = [0.8863, 0.8845, 0.8863, 0.8914, 0.8966]
hr_ml1m = [0.7151, 0.7129, 0.7195, 0.7215, 0.7175]
hr_amusic = [0.4127, 0.4127, 0.4206, 0.4313, 0.4420]

# NDCG@10指标数据
ndcg_lastfm = [0.5936, 0.6038, 0.5987, 0.6034, 0.6154]
ndcg_ml1m = [0.4351, 0.4333, 0.4333, 0.4350, 0.4335]
ndcg_amusic = [0.2467, 0.2560, 0.2479, 0.2506, 0.2542]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8), dpi=120)

# 定义颜色方案
colors = {
    'LastFM': '#0000FF',   # 更亮的蓝色
    'ML-1M': '#FFA500',    # 橙色
    'AMusic': '#008000'    # 绿色
}

# 线型和标记样式
line_styles = {
    'LastFM': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8},
    'ML-1M': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8},
    'AMusic': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8}
}

# ========== 绘制HR@10折线图 ==========
x = np.arange(len(models))

# 绘制三条折线代表三个数据集
ax.plot(x, hr_lastfm, label='LastFM', color=colors['LastFM'], **line_styles['LastFM'])
ax.plot(x, hr_ml1m, label='ML-1M', color=colors['ML-1M'], **line_styles['ML-1M'])
ax.plot(x, hr_amusic, label='AMusic', color=colors['AMusic'], **line_styles['AMusic'])

# 添加数据标签
for i, y in enumerate(hr_lastfm):
    ax.text(i, y + 0.01, f'{y:.4f}', ha='center', va='bottom', 
            fontsize=DATA_LABEL_FONT_SIZE, fontproperties=font_prop)
for i, y in enumerate(hr_ml1m):
    ax.text(i, y + 0.01, f'{y:.4f}', ha='center', va='bottom', 
            fontsize=DATA_LABEL_FONT_SIZE, fontproperties=font_prop)
for i, y in enumerate(hr_amusic):
    ax.text(i, y + 0.01, f'{y:.4f}', ha='center', va='bottom', 
            fontsize=DATA_LABEL_FONT_SIZE, fontproperties=font_prop)

# 设置纵轴范围
ax.set_ylim(0.25, 0.95)

# 设置横轴刻度和标签 - 关键修复
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)

# 单独设置横轴标签字体大小和属性 - 新添加的代码
tick_font = fm.FontProperties(
    family=font_prop.get_name(),
    size=TICK_LABEL_FONT_SIZE  # 确保字体大小被明确设置
)

# 双重保险：显式设置每个标签的字体属性
for label in ax.get_xticklabels():
    label.set_fontproperties(tick_font)
    label.set_size(TICK_LABEL_FONT_SIZE)

# 设置标题和标签
# ax.set_title('HR@10 Performance Comparison', fontsize=TITLE_FONT_SIZE, fontproperties=font_prop, pad=15)
ax.set_ylabel('HR@10', fontsize=AXIS_LABEL_FONT_SIZE, fontproperties=font_prop)
ax.grid(linestyle=':', alpha=0.6)

# 添加图例
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.5, 0.95), ncol=3,
           prop={'size': LEGEND_FONT_SIZE, 'family': font_prop.get_name()})

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.90])

# 添加底部副标题
fig.text(0.5, 0.02, "Methods: MLP (Baseline), +RAT, +RAWP, +RAWP-FT, +RAWP-PDAF", 
         ha='center', fontsize=20, fontproperties=font_prop)

# 保存图像
plt.savefig('model_comparison_linechart.png', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison_linechart.pdf', dpi=300, bbox_inches='tight')
plt.show()