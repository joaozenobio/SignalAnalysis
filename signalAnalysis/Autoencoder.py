from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.models import Model


class Autoencoder:
    def __init__(self, data=None):
        if data is not None:
            self.build_autoencoder(data)
        else:
            self.model = None

    def build_autoencoder(self, data):
        self.model = Sequential()
        self.model.add(LSTM(90, activation="sigmoid", return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
        self.model.add(LSTM(120, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(150, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(180, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(150, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(120, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(90, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(70, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(50, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(20, activation="sigmoid", name='decoder_output'))
        self.model.add(RepeatVector(data.shape[1]))
        self.model.add(LSTM(20, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(30, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(40, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(50, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(60, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(70, activation="sigmoid", return_sequences=True))
        self.model.add(Dropout(rate=0.1))
        self.model.add(LSTM(80, activation="sigmoid", return_sequences=True))
        self.model.add(LSTM(90, activation="sigmoid", return_sequences=True))
        self.model.add(TimeDistributed(Dense(data.shape[1])))
        self.model.compile(loss="mse", optimizer="adam")

    def plot_summary(self):
        print(self.model.summary())

    def plot_model(self):
        plot_model(self.model)

    def fit(self, data, epochs=10, batch_size=1, verbose=1):
        self.model.fit(x=data, y=data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def save(self, directory):
        self.model.save(f'./{directory}')

    def load(self, directory):
        self.model = load_model(f'./{directory}')

    def predict(self, data):
        encoder = Model(inputs=self.model.inputs, outputs=self.model.get_layer("encoder_output").output)
        return encoder.predict(data)
