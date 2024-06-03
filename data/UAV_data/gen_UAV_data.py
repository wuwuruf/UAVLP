# _*_ coding : utf-8 _*_
# @Time : 2024/4/19 16:32
# @Author : wfr
# @file : gen_UAV_data
# @Project : IDEA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置模拟参数
num_drones = 100
area_size = 2000
time_duration = 2 * 60 * 60  # 模拟持续时间为2小时，单位为秒
sample_interval = 10  # 每10秒采样一次
visualization_interval = 30 * 60  # 每30分钟可视化一次

# 初始化无人机位置、速度和方向
np.random.seed(0)
drone_data = {
    'ID': list(range(num_drones)),
    'x': np.full(num_drones, area_size / 2),  # 初始位置为区域中心
    'y': np.full(num_drones, area_size / 2),
    'speed': np.random.uniform(20, 50, num_drones),  # 初始速度范围[20, 50] m/s
    'direction': np.random.uniform(0, 360, num_drones)  # 初始方向范围[0, 360] degree
}

# 模拟无人机运动并采样数据
time = 0
all_data = []
while time < time_duration:
    for i in range(num_drones):

        if (time % visualization_interval == 0):
            plt.scatter(drone_data['x'][i], drone_data['y'][i])

        # 记录当前时间
        current_time = time

        # 更新无人机位置
        x_new = drone_data['x'][i] + drone_data['speed'][i] * np.cos(np.radians(drone_data['direction'][i]))
        y_new = drone_data['y'][i] + drone_data['speed'][i] * np.sin(np.radians(drone_data['direction'][i]))

        drone_data['x'][i] = max(0, min(x_new, area_size))
        drone_data['y'][i] = max(0, min(y_new, area_size))

        # 随机更新速度和方向
        drone_data['speed'][i] = np.random.uniform(20, 50)
        drone_data['direction'][i] = np.random.uniform(0, 360)

        all_data.append([drone_data['ID'][i], drone_data['x'][i], drone_data['y'][i], current_time])

    if (time % visualization_interval == 0):
        plt.xlim(0, area_size)
        plt.ylim(0, area_size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'drone_positions_{time // 60}_min.png')  # 保存图片
        plt.clf()  # 清空画布

    time += sample_interval

# 保存数据到CSV文件
data_df = pd.DataFrame(all_data, columns=['ID', 'x', 'y', 'time'])
data_df.to_csv('drone_data.csv', index=False)
