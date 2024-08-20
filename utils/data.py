import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm
import random


def select_cols(data):
    features_ = data.loc[:, [# 'DeepSpCas9',
                             'ED_pos', 'ED_len', 'type_sub', 'type_ins',
                             'type_del', 'PBS_len', 'RTT_len', 'RHA_len', 'ext_len',
                             'WTS_GC', 'PBS_GC', 'RTT_GC', 'RHA_GC', 'spacer_GC', 'ext_GC',
                             'WTS_Tm', 'PBS_Tm', 'RTT_Tm', 'RHA_Tm', 'spacer_Tm', 'ext_Tm',
                             'PBS_MFE', 'RTT_MFE', 'spacer_MFE', 'ext_MFE',
                             'ext_U', 'protoS_U']]
    target = data.loc[:, 'E']

    return features_, target


class PrimeDataset(Dataset):
    def __init(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass
