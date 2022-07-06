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

        data = pd.DataFrame()
        j = 0
        for signal in signals:
            for i in range(0, len(signal)-90, 10):
                partition = signal[i:i+90].reset_index(drop=True)
                data[j] = partition
                j += 1

        self.original_data = data

        data = np.diff(data.squeeze(), axis=0)

        data = minmax_scale(data.squeeze(), axis=0).T

        self.data = data.reshape(data.shape[0], data.shape[1], 1)
