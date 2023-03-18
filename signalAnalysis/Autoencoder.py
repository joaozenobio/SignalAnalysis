from keras.models import Sequential
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
        self.model.add(LSTM(90, return_sequences=True,
                            input_shape=(data.shape[1], data.shape[2])))
        self.model.add(LSTM(80, return_sequences=True))
        self.model.add(LSTM(70, return_sequences=True))
        self.model.add(LSTM(60, return_sequences=True))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(40, return_sequences=True))
        self.model.add(LSTM(30, return_sequences=True))
        self.model.add(LSTM(20, name='encoder_output'))
        self.model.add(RepeatVector(data.shape[1]))
        self.model.add(LSTM(20, return_sequences=True))
        self.model.add(LSTM(30, return_sequences=True))
        self.model.add(LSTM(40, return_sequences=True))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(60, return_sequences=True))
        self.model.add(LSTM(70, return_sequences=True))
        self.model.add(LSTM(80, return_sequences=True))
        self.model.add(LSTM(90, return_sequences=True))
        self.model.add(TimeDistributed(Dense(data.shape[1])))
        self.model.compile(loss="mse", optimizer="adam")

    def plot_summary(self):
        print(self.model.summary())

    def plot_model(self):
        plot_model(self.model, show_shapes=True, show_dtype=True, show_layer_names=False, expand_nested=True,
                   show_layer_activations=True)

    def fit(self, data, epochs=3, batch_size=1, verbose=1):
        self.model.fit(x=data, y=data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def save(self, directory):
        self.model.save(directory)

    def load(self, directory):
        self.model = load_model(directory)

    def predict(self, data):
        encoder = Model(inputs=self.model.inputs, outputs=self.model.get_layer("encoder_output").output)
        return encoder.predict(data)
