import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 示例数据：6个层的指标值
values = [3, 4, 5, 2, 6, 4]
n = len(values)
min_height = min(values)  # 最短板高度
radius = 1  # 木桶半径

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成每个木板的曲面
for i in range(n):
    # 计算当前木板的角度范围（每个木板占据60度）
    theta_start = np.deg2rad(i*60 - 30)
    theta_end = np.deg2rad(i*60 + 30)
    
    # 创建网格
    theta = np.linspace(theta_start, theta_end, 50)
    z = np.linspace(0, values[i], 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # 转换为笛卡尔坐标
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    
    # 绘制木板曲面
    ax.plot_surface(x, y, z_grid, color='#CD853F', alpha=0.8, edgecolor='#8B4513')

# 绘制水面（最短板决定的水平面）
theta = np.linspace(0, 2*np.pi, 100)
z = min_height
x = radius * np.cos(theta)
y = radius * np.sin(theta)
ax.plot(x, y, z, color='blue', linewidth=2, alpha=0.7)

# 添加水面填充效果
theta, r = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, radius, 20))
x_water = r * np.cos(theta)
y_water = r * np.sin(theta)
z_water = np.full_like(x_water, min_height)
ax.plot_surface(x_water, y_water, z_water, color='blue', alpha=0.2)

# 绘制桶底
theta = np.linspace(0, 2*np.pi, 100)
x_bottom = radius * np.cos(theta)
y_bottom = radius * np.sin(theta)
ax.plot(x_bottom, y_bottom, np.zeros_like(theta), color='#8B4513', linewidth=2)

# 设置视角和坐标轴
ax.view_init(elev=25, azim=-45)
ax.set_zlim(0, max(values)+1)
ax.set_axis_off()  # 隐藏坐标轴

# 添加文字说明
ax.text(0, 0, min_height*1.1, f"Capacity determined by shortest board: {min_height}", 
        color='blue', ha='center', va='center')

plt.tight_layout()
plt.show()