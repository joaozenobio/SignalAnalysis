import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import minmax_scale


class Dataset:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.videos_starting_point = []

    def dataset(self, data_directory):
        signals = []
        for file in sorted(glob.glob(f"{data_directory}/*.csv")):
            signals.append(pd.read_csv(file, index_col=0).squeeze("columns"))

        segmented_signals = []
        video_starting_point = 0
        for signal in signals:
            self.videos_starting_point.append(video_starting_point)
            for i in range(0, len(signal)-89, 10):
                partition = signal[i:i+90].reset_index(drop=True).map(abs)
                segmented_signals.append(partition)
                video_starting_point += 1
        self.videos_starting_point.append(len(segmented_signals))

        data = pd.DataFrame(segmented_signals).reset_index(drop=True)
        self.original_data = data
        data = data.diff(axis=1).drop(columns=0)
        data = minmax_scale(data.values, feature_range=(-1, 1), axis=1)
        self.data = data.reshape(data.shape[0], data.shape[1], 1)

    @staticmethod
    def process_signal(path):
        signal = pd.read_csv(path, index_col=0).squeeze("columns")

        segmented_signal = []
        for i in range(0, len(signal)-89, 10):
            partition = signal[i:i+90].reset_index(drop=True).map(abs)
            segmented_signal.append(partition)

        data = pd.DataFrame(segmented_signal).reset_index(drop=True)
        data = data.diff(axis=1).drop(columns=0)
        data = minmax_scale(data.values, feature_range=(-1, 1), axis=1)
        data = data.reshape(data.shape[0], data.shape[1], 1)
        return data


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# df = pd.read_csv("signals/result_JN0PFE9.csv", index_col=0).squeeze("columns")
# fig, ax = plt.subplots(figsize=(15, 15))
# ax.plot(df.map(abs))
# cmap = plt.cm.get_cmap("hsv", len(list(range(0, 250-180, 10)))+1)
# print(cmap(1))
# for i in range(0, 250-89, 50):
#     color = np.random.rand(3,)
#     rect = patches.Rectangle((i, 6), 90, 7, linewidth=3, edgecolor=color, facecolor='none', zorder=i)
#     ax.add_patch(rect)
# plt.savefig(f'report/result_JN0PFE9.png', dpi=100)
