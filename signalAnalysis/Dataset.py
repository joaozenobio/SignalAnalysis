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

    def dataset_teste2(self, data_directory):
        signals = []
        signals_audio_name = []
        for file in sorted(glob.glob(f"{data_directory}/*.csv")):
            signals.append(pd.read_csv(file, index_col=0).squeeze("columns"))
            signals_audio_name.append(file.split('/')[-1])
        data = pd.DataFrame(signals, index=signals_audio_name).fillna(0)
        data['label'] = [label.split('-')[4] for label in signals_audio_name]
        data.to_csv('original_data.csv')
        data = data.drop(columns=['label'])
        data = minmax_scale(data.values, feature_range=(0, 1), axis=1)
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

    def dataset_teste_grande(self, data_directory):
        signals = []
        signals_audio_name = []
        for file in sorted(glob.glob(f"{data_directory}/*.csv")):
            signals.append(pd.read_csv(file, index_col=0).squeeze("columns"))
            signals_audio_name.append(file.split('/')[-1])
        data = pd.DataFrame(signals, index=signals_audio_name).fillna(0)
        data['label'] = [label.split('_')[0] for label in signals_audio_name]
        data.to_csv('original_data.csv')
        data = data.drop(columns=['label'])
        data = minmax_scale(data.values, feature_range=(0, 1), axis=1)
        self.data = data.reshape(data.shape[0], data.shape[1], 1)

import matplotlib.pyplot as plt
import matplotlib.patches as patches


df = pd.read_csv("old/signals_old/result_5YzozaP.csv", index_col=0).squeeze("columns").map(abs)[0:91]
df = df.diff()
df = pd.DataFrame(minmax_scale(df.values, feature_range=(-1, 1)))[1:]
fig, ax = plt.subplots(figsize=(15, 15))
ax.plot(df, linewidth=3)
# for i in range(0, len(df)-89, 50):
#     color = np.random.rand(3,)
#     rect = patches.Rectangle((i, min(df)), 90, max(df)-min(df), linewidth=3, edgecolor=color, facecolor='none', zorder=i)
#     ax.add_patch(rect)
plt.savefig(f'old/signals_old/sinal_processado.jpg', dpi=100)
