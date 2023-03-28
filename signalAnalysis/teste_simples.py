import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os


def teste():
    os.makedirs("./signals", exist_ok=True)

    for i in range(50):
        x = np.linspace(-np.pi, np.pi, 91)
        noise = np.random.default_rng().normal(1, 0.5, size=(len(x),)) / 100
        wave = np.sin(x) + noise
        wave = wave / max(wave)
        pd.DataFrame(wave).to_csv(f"signals/wave_1_{i}.csv")

    plt.plot(x, wave)
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    plt.show()

    for i in range(50):
        x = np.linspace(-np.pi, np.pi, 91)
        noise = np.random.default_rng().normal(1, 0.5, size=(len(x),)) / 100
        wave = np.linspace(0.5, 1, 91) + noise
        wave = wave / max(wave)
        pd.DataFrame(wave).to_csv(f"signals/wave_2_{i}.csv")

    plt.plot(x, wave)
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    plt.show()
