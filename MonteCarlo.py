import numpy as np
import matplotlib.pyplot as plt

# ====================== 参数设置 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 基础参数
M = 1000  # 区域电动自行车总数
N = 500  # 蒙特卡洛仿真次数
time_resolution = 1440  # 时间轴分段数(24小时×60分钟)
#days = 1  # 模拟天数

# 电池参数
E1 = 0.96  # 电池容量(kWh)
η = 0.9  # 充电效率
P1 = 0.165  # 充电功率(kW)
energy_consume = 0.02  # 能耗(kWh/km)

# 行驶里程分布(对数正态分布参数)
mu = 1.8  # 对数均值(对应实际里程约6km)
sigma = 0.6  # 对数标准差

# 充电开始时间分布(正态分布参数)
start_mean = 18 * 60  # 均值18:00(分钟)
start_std = 90  # 标准差90分钟


# ====================== 蒙特卡洛仿真核心 ======================
def monte_carlo_simulation():
    total_load = np.zeros(time_resolution)

    for _ in range(N):
        # 单次仿真负荷记录
        sim_load = np.zeros(time_resolution)

        # 生成日行驶里程(对数正态分布)
        daily_mileage = np.random.lognormal(mu, sigma, M)
        daily_mileage = np.clip(daily_mileage, 0, 100)  # 限制合理范围

        # 计算SOC(基于物理模型)
        energy_used = daily_mileage * energy_consume
        SOC = (E1 - energy_used) / E1
        SOC = np.clip(SOC, 0, 1)  # 电量范围约束

        # 生成充电开始时间(分钟)
        start_times = np.random.normal(start_mean, start_std, M)
        start_times = np.clip(start_times, 0, 1439).astype(int)

        # 计算充电时长(分钟)
        charge_hours = (E1 * (1 - SOC)) / (η * P1)  # 理论小时数
        charge_duration = np.ceil(charge_hours * 60).astype(int)  # 转换为分钟

        # 负荷叠加
        for i in range(M):
            if SOC[i] >= 1:  # 无需充电
                continue

            start = start_times[i]
            duration = charge_duration[i]
            end = start + duration

            # 处理跨天充电
            if end > time_resolution:
                # 前半夜充电部分
                sim_load[start:] += P1
                # 后半夜充电部分
                overflow = end - time_resolution
                sim_load[:overflow] += P1
            else:
                sim_load[start:end] += P1

        total_load += sim_load

    return total_load / N  # 返回平均负荷

# ====================== 执行仿真 ======================
average_load = monte_carlo_simulation()
# 将分钟级数据转换为小时级平均值
hourly_load = average_load.reshape(24, 60).mean(axis=1)
# 修改输出部分代码
print("\n时间\t\t负荷(kW)")
print("----------------------")
for minute in range(60, 1441, 60):  # 每60分钟输出一次
    hour = minute // 60
    time_str = f"{hour}:00" if hour <= 24 else "0:00"
    print(f"{time_str}\t{average_load[minute-1]:.2f}")  # 取每小时最后1分钟的值

# ====================== 可视化结果 ======================
plt.figure(figsize=(12, 6))
time_axis = np.linspace(0, 24, time_resolution, endpoint=False)
plt.plot(time_axis, average_load, color='darkorange', linewidth=1.2)
plt.fill_between(time_axis, average_load, alpha=0.2, color='gold')

plt.title('电动自行车充电负荷时间分布预测', fontsize=14)
plt.xlabel('时间 (小时)', fontsize=12)
plt.ylabel('充电功率 (kW)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 1))
plt.tight_layout()
plt.show()

# ====================== 充电桩数量计算 ======================
peak_load = np.max(average_load)
required_chargers = np.ceil(peak_load / P1).astype(int)

print("\n" + "=" * 45)
print(f"■ 峰值充电功率: {peak_load:.2f} kW")
print(f"■ 建议充电桩配置: {required_chargers} 个 (按峰值需求)")
#print(f"■ 平均利用率: {(np.mean(average_load) / peak_load * 100):.1f}%")
print("=" * 45)