import glob
import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.io import wavfile


def teste_grande(dataset_path):
    os.makedirs("./signals", exist_ok=True)
    os.makedirs("./new_audio", exist_ok=True)
    for audio in sorted(os.listdir(dataset_path)):
        wave = AudioSegment.from_wav(f'{dataset_path}/{audio}')
        wave = wave.set_frame_rate(8000)
        wave.export(f'./new_audio/{audio}', format="wav")
        sample_rate, wave = wavfile.read(f'./new_audio/{audio}')
        pd.DataFrame(wave).to_csv(
            f'./signals/{audio.replace(".wav", ".csv")}'
        )

