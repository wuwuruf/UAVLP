# _*_ coding : utf-8 _*_
# @Time : 2024/5/13 15:39
# @Author : wfr
# @file : gen_UAV_SNR
# @Project : IDEA

import numpy as np
import pandas as pd
#
# # 设置随机种子，以便结果可重复
# np.random.seed(0)
#
# # 节点数量
# n = 100
#
# # 生成节点对，注意(i < j)
# nodes = [(i, j) for i in range(n) for j in range(i+1, n)]
#
# # 生成正态分布信噪比
# snr = np.random.normal(loc=17.5, scale=3, size=len(nodes))  # 均值为17.5，标准差为3
#
# # 限制信噪比范围在10~25dB之间
# snr = np.clip(snr, 10, 25)
#
# # 创建DataFrame
# df = pd.DataFrame({'i': [pair[0] for pair in nodes],
#                    'j': [pair[1] for pair in nodes],
#                    'SNR': snr})
#
# # 保存到CSV文件
# df.to_csv('snr_data.csv', index=False)


# 读取 txt 文件，指定空格为分隔符，设置表头
df = pd.read_csv('UAV_RPGM_data.txt', sep='\s+', header=0)

# 修改列名
df.columns = ['ID', 'time', 'x', 'y']

# 将 DataFrame 保存为 CSV 文件，指定逗号为分隔符
df.to_csv('UAV_RPGM_data.csv', index=False, sep=',')
