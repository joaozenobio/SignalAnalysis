import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import minmax_scale
import warnings
warnings.filterwarnings('ignore')


class Dataset:
    def __init__(self, data_directory):
        signals = []
        for file in glob.glob(f"{data_directory}/*.csv"):
            signals.append(pd.read_csv(file, index_col=0).squeeze("columns"))
        signals.append(pd.read_csv("result_aJdKrfM.csv", index_col=0).squeeze("columns"))

        segmented_signals = []
        for signal in signals:
            for i in range(0, len(signal)-89, 10):
                partition = signal[i:i+90].reset_index(drop=True)
                segmented_signals.append(partition)

        data = pd.DataFrame(segmented_signals).reset_index(drop=True)

        self.original_data = data

        data = data.diff(axis=1).drop(columns=0)

        data = minmax_scale(data.values, feature_range=(-1, 1), axis=1)

        self.data = data.reshape(data.shape[0], data.shape[1], 1)
