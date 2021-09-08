import math
import numbers

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import pandas as pd

import plotly.express as px
from plotly import graph_objects as go


def create_sequences(df, metrics_column, time_steps):
    x = df[[metrics_column]]
    y = df[metrics_column]
    xs = []
    ys = []
    for i in range(len(x) - time_steps):
        xs.append(x.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(xs), np.array(ys)


def find_batch_size(time_steps):
    return math.ceil(time_steps / 8) * 8


def find_time_steps(df, datetime_column):
    if df[datetime_column].dtype.type is not np.datetime64:
        raise TypeError('datetime_column must be of type <np.datetime64>')

    delta = df[datetime_column].diff().median()

    if delta >= pd.Timedelta('7 days'):
        return 30

    if delta >= pd.Timedelta('1 day'):
        return 7

    if delta >= pd.Timedelta('1 hour'):
        return 24

    if delta >= pd.Timedelta('1 minute'):
        return 60

    if delta >= pd.Timedelta('1 second'):
        return 60

    return 10


def find_datetime_column(df):
    for column in df:
        if df[column].dtype.type is np.datetime64:
            return column

    return None


def prep_datetime(df, datetime_column):
    df = df.copy()
    if df[datetime_column].dtype.type is not np.datetime64:
        df[datetime_column] = pd.to_datetime(df[datetime_column])

    df = df.sort_values(by=datetime_column, ascending=True)
    return df


# heavily inspired by https://github.com/susanli2016/Machine-Learning-with-Python
class TimeSeriesAnomalyDetector:

    def __init__(self, df, metrics_column, datetime_column=None, ScalerClass=None, time_steps=None, split=0.8):
        self.metrics_column = metrics_column

        self.datetime_column = datetime_column
        if datetime_column is None:
            self.datetime_column = find_datetime_column(df)
            if self.datetime_column is None:
                raise Exception('Cannot find a datetime_column, please supply one')

        self.df = prep_datetime(df, self.datetime_column)

        self.time_steps = time_steps
        if time_steps is None:
            self.time_steps = find_time_steps(self.df, self.datetime_column)

        self.split = split
        self.ScalerClass = ScalerClass
        self.score_df = None
        self.sample()

    def sample(self, split=None):
        # split data into training and testing samples
        if split is None:
            split = self.split

        mask = np.random.rand(len(self.df)) < split
        self.train = self.df[mask].copy().reset_index(drop=True)
        self.test = self.df[~mask].copy().reset_index(drop=True)

        self.scale()  # no-op if self.ScalerClass is None

    def scale(self, ScalerClass=None):
        if ScalerClass is None:
            ScalerClass = self.ScalerClass

        if ScalerClass:
            scaler = ScalerClass()
            scaler = scaler.fit(self.train[[self.metrics_column]])
            self.train[self.metrics_column] = scaler.transform(self.train[[self.metrics_column]])
            self.test[self.metrics_column] = scaler.transform(self.test[[self.metrics_column]])

        self.sequence()

    def sequence(self, time_steps=None):
        if time_steps is None:
            time_steps = self.time_steps
        self.x_train, self.y_train = create_sequences(self.train, self.metrics_column, time_steps)
        self.x_test, self.y_test = create_sequences(self.test, self.metrics_column, time_steps)

    def create_model(self, verbose=True):
        # model layer, configure automatically?
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(self.x_train.shape[1]))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.x_train.shape[2])))
        self.model.compile(optimizer='adam', loss='mae')
        if verbose:
            self.model.summary()
        return self.model

    def train_model(self, **fit_params):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=100,
            batch_size=find_batch_size(self.time_steps),
            validation_split=1 - self.split,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
            shuffle=False,
            **fit_params,
        )
        return self.model

    def evaluate_model(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def predict(self, df=None, percentile=None):
        if df is None:
            df = self.df

            if self.ScalerClass:
                scaler = self.ScalerClass()
                scaler = scaler.fit(df[[self.metrics_column]])
                df[self.metrics_column] = scaler.transform(df[[self.metrics_column]])

        x, _ = create_sequences(df, self.metrics_column, self.time_steps)
        x_pred = self.model.predict(x, verbose=0)
        mae_loss = np.mean(np.abs(x_pred - x), axis=1)  # TODO: make configurable?

        if not percentile or not isinstance(percentile, numbers.Number):
            percentile = 99

        threshold = np.percentile(mae_loss, percentile)

        score_df = pd.DataFrame(df[self.time_steps:])
        score_df['loss'] = mae_loss
        score_df['threshold'] = threshold
        score_df['anomaly'] = score_df['loss'] > score_df['threshold']
        self.score_df = score_df
        return score_df

    def get_anomalies(self):
        if self.score_df is not None:
            return self.score_df.loc[self.score_df['anomaly'] == True]

    def graph_loss(self):
        if self.score_df is not None:
            fig = px.histogram(self.score_df['loss'])
            return fig

    def graph_loss_threshold(self):
        if self.score_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.score_df[self.datetime_column], y=self.score_df['loss'], name='Test loss'))
            fig.add_trace(go.Scatter(x=self.score_df[self.datetime_column], y=self.score_df['threshold'], name='Threshold'))
            fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
            return fig

    def graph(self):
        anomalies = self.get_anomalies()
        score_y = self.score_df[self.metrics_column]
        anom_y = anomalies[self.metrics_column]
        if self.ScalerClass:
            scaler = self.ScalerClass()
            scaler = scaler.fit(self.df[[self.metrics_column]])
            score_y = scaler.inverse_transform(score_y)
            anom_y = scaler.inverse_transform(anom_y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.score_df[self.datetime_column], y=score_y, name=self.metrics_column))
        fig.add_trace(go.Scatter(x=anomalies[self.datetime_column], y=anom_y, mode='markers', name='Anomaly'))
        fig.update_layout(showlegend=True, title='Detected anomalies')
        return fig
