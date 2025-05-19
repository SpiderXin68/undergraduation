import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Voronoi, voronoi_plot_2d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 学生公寓坐标数据 - 可自由修改
facility_points = np.array([
[2956.48, 2292.41],
[2425.52, 2292.41],
[2255.84, 1695.77],
[1801.52, 1876.40],
[1330.77, 1969.46],
[1325.30, 2407.36],
[1626.36, 2823.36],
[3941.75, 4607.80],
[3476.48, 4602.33],
[2983.85, 4175.38],
[3011.22, 4569.49],
[2507.63, 4153.48],
[2524.05, 4558.54],
[2737.53, 5062.13]
])

# 各设施点需求数据（新增）
demand = np.array([
    75, 63, 83, 72, 74,
    80, 72, 89, 70, 73,
    71, 78, 85, 48
])
# 数据标准化
scaler = StandardScaler()
facility_points_scaled = scaler.fit_transform(facility_points)

# 动态确定聚类数量（不超过点数且至少为1）
n_clusters = min(8, len(facility_points))

# K-means聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(facility_points_scaled)
labels = kmeans.labels_  # 获取每个点的聚类标签

# 获取聚类中心并反标准化
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# 打印充电站信息
print("=== 充电站选址及服务范围 ===")
for i in range(n_clusters):
    # 找出属于当前聚类的所有点
    cluster_points = facility_points[labels == i]
    cluster_demands = demand[labels == i]
    point_indices = np.where(labels == i)[0] + 1  # 公寓编号从1开始

    print(f"\n充电站 {i + 1}:")
    print(f"坐标: ({cluster_centers[i, 0]:.2f}, {cluster_centers[i, 1]:.2f})")
    print(f"服务点: {point_indices.tolist()}")
    print(f"服务点数量: {len(cluster_points)}个")
    print(f"总需求量: {sum(cluster_demands)}辆/天")
    print("详细需求分布:")
    for idx, d in zip(point_indices, cluster_demands):
        print(f"  公寓{idx}: {d}辆/天")

# 设置画布大小（动态调整）
fig_size = max(10, len(facility_points) / 3)
plt.figure(figsize=(fig_size, fig_size))
ax = plt.gca()

# 动态计算坐标轴范围
x_min, x_max = facility_points[:, 0].min(), facility_points[:, 0].max()
y_min, y_max = facility_points[:, 1].min(), facility_points[:, 1].max()
margin = max(x_max - x_min, y_max - y_min) * 0.2  # 20%边距
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_aspect('equal')

# 为每个聚类分配颜色
#colors = plt.cm.get_cmap('tab10', n_clusters)
colors=plt.get_cmap('tab10', n_clusters)


# 绘制需求点（按聚类着色）
scatter = ax.scatter(
    facility_points[:, 0], facility_points[:, 1],
    c=labels, cmap='tab10', s=np.sqrt(demand) * 10,
    edgecolors='k', alpha=0.9, zorder=4
)

# 绘制聚类中心
ax.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c=range(n_clusters), cmap='tab10', marker='X',
    s=400, edgecolors='k', linewidth=2,
    label='充电站', zorder=5
)

# 绘制连接线（显示服务关系）
for i in range(len(facility_points)):
    station_idx = labels[i]
    ax.plot(
        [facility_points[i, 0], cluster_centers[station_idx, 0]],
        [facility_points[i, 1], cluster_centers[station_idx, 1]],
        color=colors(station_idx), alpha=0.3,
        linestyle='--', zorder=3
    )

# 添加标签（公寓点）
for i, (x, y) in enumerate(facility_points):
    ax.text(x, y + margin * 0.03,
            f'点{i + 1}',
            fontsize=9, ha='center',
            bbox=dict(facecolor='white', alpha=0.8))

# 为充电站添加标签
for i, (x, y) in enumerate(cluster_centers):
    ax.text(x, y - margin * 0.05,
            f'充电站{i + 1}',
            fontsize=11, ha='center',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.9))

# 绘制Voronoi图（只有多于1个聚类中心时才绘制）
if n_clusters > 1:
    vor = Voronoi(cluster_centers)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                    line_colors='orange', line_alpha=0.6,
                    show_points=False)
    # 再次锁定坐标轴范围
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

# 添加比例尺（动态调整位置和长度）
scale_length = round((x_max - x_min) / 5, -2)  # 取坐标范围的1/5，整百数
scale_x = x_max - margin * 0.5
scale_y = y_min + margin * 0.2
ax.plot([scale_x, scale_x + scale_length],
        [scale_y, scale_y], color='black', lw=2)
ax.text(scale_x + scale_length / 2, scale_y + margin * 0.02,
        f'{int(scale_length)}米', ha='center', fontsize=10)

# 添加图例和标题
ax.set_title(f'校园电动自行车充电站选址（共{len(facility_points)}个需求点,{len(cluster_centers)}个充电站）',
             fontsize=18, pad=20)
ax.set_xlabel('X坐标（米）', fontsize=12)
ax.set_ylabel('Y坐标（米）', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)

# 添加颜色条
cbar = plt.colorbar(scatter, ticks=range(n_clusters))
cbar.ax.set_yticklabels([f'充电站{i + 1}' for i in range(n_clusters)])
cbar.set_label('服务区域', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()