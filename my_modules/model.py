# _*_ coding : utf-8 _*_
# @Time : 2024/6/5 12:50
# @Author : wfr
# @file : model
# @Project : IDEA

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiAggLP(Module):
    """

    """
