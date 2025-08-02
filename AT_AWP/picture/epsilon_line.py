# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation

# 设置横轴数据
x_values = [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100]
x_values = x_values[5:-3]
# RAWP - RAWP-FT - FTS
y_values_1 = [72.04,72.12,72.30,71.26,66.13,62.67, 53.97, 49.67, 40.68, 35.68, 31.44, 28.91, 27.09, 25.94]
y_values_2 = [71.94,71.87,71.95,72.09,70.89,69.27, 61.97, 58.15, 49.57, 44.11, 38.16, 34.19, 32.68, 30.31]
y_values_3 = [71.71,71.57,71.54,71.11,68.48,67.45, 64.27, 62.27, 56.52, 51.06, 43.94, 35.70, 31.23, 23.43]
y_values_4 = [43.36,43.41,43.23,41.97,36.95,34.52, 28.52, 25.85, 21.16, 18.60, 16.56, 14.82, 13.71, 13.44]
y_values_5 = [43.50,43.47,43.44,42.66,39.90,38.04, 32.97, 30.21, 25.22, 22.36, 19.51, 17.53, 16.42, 15.26]
y_values_6 = [42.85,42.86,42.74,42.01,39.82,38.77, 35.89, 34.00, 29.54, 26.03, 21.91, 17.63, 14.90, 10.72]

y_values_1 = y_values_1[5:-3]
y_values_2 = y_values_2[5:-3]
y_values_3 = y_values_3[5:-3]
y_values_4 = y_values_4[5:-3]
y_values_5 = y_values_5[5:-3]
y_values_6 = y_values_6[5:-3]

# 创建图形和子图
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 设置主标题
fig.suptitle('HR@10 and NDCG@10 for RAWP、RAWP-FT and PDAF Models', 
             fontsize=14, fontweight='bold', y=1.05)

# # 配置对数刻度
# for ax in axes:
#     ax.set_xscale('log')
#     ax.set_xlim(0.004, 0.04)  # 设置x轴范围
#     ax.set_xlabel('Epsilon', fontsize=12)  # 使用LaTeX格式
#     ax.xaxis.set_major_formatter(ScalarFormatter())
#     ax.set_xticks([0.005, 0.008, 0.010, 0.015, 0.020, 0.030])
#     ax.set_xticklabels([0.005, 0.008, 0.010, 0.015, 0.020, 0.030], 
#                       rotation=45, ha='right')  # 旋转45度
#     ax.grid(True, linestyle='--', alpha=0.5)

# 配置对数刻度
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlim(0.0045, 0.032)
    ax.set_xlabel('Epsilon', fontsize=12)
    
    # 精准控制主刻度
    ax.set_xticks([0.005, 0.01, 0.02])
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(plt.NullLocator())  # 关键：禁用次要刻度
    
    # 设置刻度标签
    ax.set_xticklabels([
        r'$5 \times 10^{-3}$',
        r'$1 \times 10^{-2}$',
        r'$2 \times 10^{-2}$'
    ], rotation=0, ha='right', fontsize=10)
    
    # 添加垂直虚线
    for eps in [0.005, 0.006, 0.007, 0.008, 0.01, 0.01167, 0.01335, 0.015, 0.0175, 0.02, 0.023, 0.026, 0.0285, 0.03]:
        ax.axvline(x=eps, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
    
    ax.grid(True, linestyle='--', alpha=0.5)

# 绘制HR@10 (使用虚线连接)
axes[0].plot(x_values, y_values_1, marker='o', markersize=4, linestyle='--',
            label='RAWP-MLP', color='tab:blue', linewidth=1)
axes[0].plot(x_values, y_values_2, marker='o', markersize=4, linestyle='--',
            label='RAWP-FT-MLP', color='tab:orange', linewidth=1)
axes[0].plot(x_values, y_values_3, marker='o', markersize=4, linestyle='--',
            label='PDAF-MLP', color='tab:green', linewidth=1)
axes[0].set_title("HR@10 for RAWP, RAWP-FT and PDAF")
axes[0].set_ylabel('HR@10 (%)', fontsize=12)
axes[0].set_ylim(30, 70)
axes[0].legend(loc='lower left', frameon=True)

# 绘制NDCG@10 (使用虚线连接)
axes[1].plot(x_values, y_values_4, marker='o', markersize=4, linestyle='--',
            label='RAWP-MLP', color='tab:blue', linewidth=1)
axes[1].plot(x_values, y_values_5, marker='o', markersize=4, linestyle='--',
            label='RAWP-FT-MLP', color='tab:orange', linewidth=1)
axes[1].plot(x_values, y_values_6, marker='o', markersize=4, linestyle='--',
            label='PDAF-MLP', color='tab:green', linewidth=1)
axes[1].set_title("NDCG@10 for RAWP, RAWP-FT and PDAF")
axes[1].set_ylabel('NDCG@10 (%)', fontsize=12)
axes[1].set_ylim(15, 40)
axes[1].legend(loc='lower left', frameon=True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('robust_performance.png', dpi=300, bbox_inches='tight')
plt.show()