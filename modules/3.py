# _*_ coding : utf-8 _*_
# @Time : 2024/6/20 15:33
# @Author : wfr
# @file : 3
# @Project : IDEA

# win_size = 10
# theta = 0.5
# for i in range(win_size):
#     decay = (1 - theta) ** (win_size - i - 1)
#     print(decay)
import torch

features = torch.randn(100, 32)
print(1)