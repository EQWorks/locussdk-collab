import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

import math


np.random.seed(1)
tf.random.set_seed(1)


def create_sequences(df, metrics_column, time_steps):
    x = df[[metrics_column]]
    y = df[metrics_column]
    xs = []
    ys = []
    for i in range(len(x) - time_steps):
        xs.append(x.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(xs), np.array(ys)


def find_batch_size(timesteps):
    return math.ceil(timesteps / 8) * 8


class AnomalyDetector:

    def __init__(self, df, metrics_column, ScalerClass=StandardScaler, time_steps=7, split=0.8):
        self.df = df.copy()
        self.metrics_column = metrics_column
        self.ScalerClass = ScalerClass
        self.time_steps = time_steps
        self.split = split
        self.scaled = False

    def split_samples(self):
        mask = np.random.rand(len(df)) < self.split
        self.train = df[mask].copy().reset_index(drop=True)
        self.test = df[~mask].copy().reset_index(drop=True)

    def scale_samples(self):
        scaler = self.ScalerClass()
        scaler = scaler.fit(self.train[[self.metrics_column]])
        self.train[self.metrics_column] = scaler.transform(self.train[[self.metrics_column]])
        self.test[self.metrics_column] = scaler.transform(self.test[[self.metrics_column]])
        self.scaled = True

    def generate_sequences(self):
        self.x_train, self.y_train = create_sequences(train, self.metrics_column, self.time_steps)
        self.x_test, self.y_test = create_sequences(test, self.metrics_column, self.time_steps)

    def create_model(self, verbose=True):
        # model layer, configure automatically?
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(x_train.shape[1]))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(x_train.shape[2])))
        self.model.compile(optimizer='adam', loss='mae')
        if verbose:
            self.model.summary()
        return self.model

    def train_model(self):
        # TODO: allow fit params config
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=100,
            batch_size=find_batch_size(self.time_steps),
            validation_split=1 - self.split,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
            shuffle=False,
        )
        return self.model

    def evaluate_model(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, df):
        scaler = StandardScaler()
        scaler = scaler.fit(df[[self.metrics_column]])

        if self.scaled:
            df[self.metrics_column] = scaler.transform(df[[self.metrics_column]])

        x, y = create_sequences(df, self.metrics_column, self.time_steps)
        x_pred = model.predict(x, verbose=0)
        mae_loss = np.mean(np.abs(x_pred - x), axis=1)  # TODO: make configurable?

        # TODO: std find outlier as threshold
        threshold = np.percentile(mae_loss, 99)  # best way to pick this threshold?

        score_df = pd.DataFrame(df[self.time_steps:])
        score_df['loss'] = mae_loss
        score_df['threshold'] = threshold
        score_df['anomaly'] = score_df['loss'] > score_df['threshold']
        return score_df
