import pandas as pd
from utils.data import select_cols

train = pd.read_csv('/Users/yousefjan/PycharmProjects/peg_nn/processed_data/train.csv')

features, _ = select_cols(train)

mean, std = features.mean(), features.std()

mean.to_csv('/Users/yousefjan/PycharmProjects/peg_nn/processed_data/mean.csv',
            header=False)
std.to_csv('/Users/yousefjan/PycharmProjects/peg_nn/processed_data/std.csv',
           header=False)

