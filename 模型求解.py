import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False


# ====================== 蒙特卡洛充电需求预测函数 ======================
def predict_chargers(M, N=500):
    """蒙特卡洛仿真预测充电桩需求
    Parameters:
        M (int): 电动自行车数量
        N (int): 仿真次数，默认500次

    Returns:
        int: 建议充电桩数量
    """
    # 参数设置（保持不变）
    time_resolution = 1440
    E1 = 0.96
    η = 0.9
    P1 = 0.165
    energy_consume = 0.02
    mu = 1.8
    sigma = 0.6
    start_mean = 18 * 60
    start_std = 90

    total_load = np.zeros(time_resolution)

    for _ in range(N):
        sim_load = np.zeros(time_resolution)

        daily_mileage = np.random.lognormal(mu, sigma, M)
        daily_mileage = np.clip(daily_mileage, 0, 100)

        energy_used = daily_mileage * energy_consume
        SOC = (E1 - energy_used) / E1
        SOC = np.clip(SOC, 0, 1)

        start_times = np.random.normal(start_mean, start_std, M)
        start_times = np.clip(start_times, 0, 1439).astype(int)

        charge_hours = (E1 * (1 - SOC)) / (η * P1)
        charge_duration = np.ceil(charge_hours * 60).astype(int)

        for i in range(M):
            if SOC[i] >= 1:
                continue

            start = start_times[i]
            duration = charge_duration[i]
            end = start + duration

            if end > time_resolution:
                sim_load[start:] += P1
                overflow = end - time_resolution
                sim_load[:overflow] += P1
            else:
                sim_load[start:end] += P1

        total_load += sim_load

    average_load = total_load / N
    peak_load = np.max(average_load)
    return int(np.ceil(peak_load / P1))


# ====================== 校园充电站规划主程序 ======================
# 成本参数（保持不变）
C_f = 15000
C_p = 800
C_d = 0.8
C_plan =16

# 校园建筑坐标与需求数据
facility_points = np.array([
    [2956.48, 2292.41], [2425.52, 2292.41], [2255.84, 1695.77],
    [1801.52, 1876.40], [1330.77, 1969.46], [1325.30, 2407.36],
    [1626.36, 2823.36], [3941.75, 4607.80], [3476.48, 4602.33],
    [2983.85, 4175.38], [3011.22, 4569.49], [2507.63, 4153.48],
    [2524.05, 4558.54], [2737.53, 5062.13]
])

# 各设施点需求数据（新增）
demand = np.array([
    75, 63, 83, 72, 74,
    80, 72, 89, 70, 73,
    71, 78, 85, 48
])

# 全局参数
total_e_bikes = 1000
total_demand = demand.sum()  # 改为总需求计算
results = []

# 遍历k值范围3-12
for k in range(3, 13):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(facility_points)

    chargers = []
    cluster_distances = []

    for j in range(k):
        mask = (labels == j)

        # 修改点：按实际需求分配电动自行车
        cluster_demand = demand[mask].sum()  # 获取当前聚类总需求
        e_bikes = int(total_e_bikes * cluster_demand / total_demand)

        required_chargers = predict_chargers(e_bikes)
        chargers.append(required_chargers)

        center = kmeans.cluster_centers_[j]
        dist = np.sum(distance.cdist(facility_points[mask], [center], 'euclidean'))
        cluster_distances.append(dist)

    # 成本计算（保持不变）
    total_distance = sum(cluster_distances)
    cost_components = {
        'k': k,
        'chargers': chargers,
        'fixed': C_f * k,
        'variable': C_p * sum(chargers),
        'distance': C_d * total_distance,
        'planning': C_plan * sum([n ** 2 for n in chargers])
    }
    cost_components['total'] = sum([cost_components[k] for k in ['fixed', 'variable', 'distance', 'planning']])

    results.append(cost_components)

# ====================== 结果输出优化 ======================
# 打印结果表格（优化格式）
headers = ["k值", "充电桩分布", "固定成本", "可变成本", "距离成本", "规划成本", "总成本"]
col_widths = [6, 25, 10, 10, 10, 10, 12]

# 打印表头
header_line = "|".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
print(f"\n{' 充电站规划结果分析 ':=^{len(header_line)}}")
print("=" * len(header_line))
print(header_line)
print("=" * len(header_line))

# 打印数据行
for res in results:
    charger_str = ",".join(map(str, res['chargers']))
    row = [
        str(res['k']),
        charger_str,
        f"{res['fixed']}",
        f"{res['variable']}",
        f"{res['distance']:.0f}",
        f"{res['planning']}",
        f"{res['total']}"
    ]
    row_line = "|".join(f"{cell:^{w}}" for cell, w in zip(row, col_widths))
    print(row_line)

print("=" * len(header_line))

# 可视化优化
plt.figure(figsize=(12, 6))
plt.plot([res['k'] for res in results],
         [res['total'] / 10000 for res in results],  # 转换为万元
         's-', color='#2ca02c', linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2)

plt.title('充电站数量与总成本关系（单位：万元）', fontsize=14)
plt.xlabel('充电站数量 (k)', fontsize=12)
plt.ylabel('总成本 (万元)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(3, 13))
plt.gca().yaxis.set_major_formatter('{x:.1f}万')  # Y轴显示为万元

# 添加数据标签
for res in results:
    plt.text(res['k'], res['total'] / 10000 + 0.3,
             f"{res['total'] / 10000:.1f}",
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()