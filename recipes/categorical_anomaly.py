from typing import Union
import json

import pandas as pd
from sklearn.ensemble import IsolationForest
import geopandas as gpd

import plotly.express as px


class CategoricalAnomalyDetector:

    def __init__(
        self,
        df: Union[pd.DataFrame, gpd.GeoDataFrame],
        metrics_column: str,
        category_column: str,
        ScalerClass = None,
    ):
        self.df = df.copy()
        self.metrics_column = metrics_column
        self.category_column = category_column
        self.ScalerClass = ScalerClass
        self.score_df = None
        self.scale()

    def scale(self, ScalerClass = None):
        if ScalerClass is None:
            ScalerClass = self.ScalerClass

        if ScalerClass:
            scaler = ScalerClass()
            scaler = scaler.fit(self.df[[self.metrics_column]])
            self.df[self.metrics_column] = scaler.transform(self.df[[self.metrics_column]])

    def create_model(self):
        self.model = IsolationForest()
        return self.model

    def predict(self, df = None):
        if df is None:
            df = self.df

            if self.ScalerClass:
                scaler = self.ScalerClass()
                scaler = scaler.fit(df[[self.metrics_column]])
                df[self.metrics_column] = scaler.transform(df[[self.metrics_column]])

        score_df = df.copy()
        score_df['score'] = self.model.fit_predict(score_df[[self.metrics_column]])
        score_df['anomaly'] = score_df['score'] <= -1
        self.score_df = score_df
        return score_df

    def get_anomalies(self):
        if self.score_df is not None:
            return self.score_df.loc[self.score_df['anomaly'] == True]

    def graph(self, no_geo: bool = False, **kwargs):
        if no_geo or not isinstance(self.score_df, gpd.GeoDataFrame):
            plotly_args = {
                'data_frame': self.score_df,
                'x': self.category_column,
                'y': self.metrics_column,
                'color': 'anomaly',
                'symbol': 'anomaly',
                **kwargs,
            }
            return px.scatter(**plotly_args)

        plotly_args = {
            'data_frame': self.score_df,
            'geojson': json.loads(self.score_df.to_json()),
            'featureidkey': f'properties.{self.category_column}',
            'locations': self.category_column,
            'color': 'anomaly',
            'hover_name': self.metrics_column,
            'opacity': 0.5,
            'mapbox_style': 'carto-positron',
            'center': {'lat': 44, 'lon': -79},
            'zoom': 7,
            'height': 720,
            **kwargs,
        }
        return px.choropleth_mapbox(**plotly_args)
