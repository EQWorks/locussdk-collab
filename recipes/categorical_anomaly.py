from typing import Union
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import tqdm

from collections import defaultdict

import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from anlearn.loda import LODA
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import nct

class CategoricalAnomalyDetector:

    def __init__(
        self,
        df: Union[pd.DataFrame, gpd.GeoDataFrame],
        metrics_column: str,
        category_column: str,
        model_type: str,
        ScalerClass = None,
    ):
        self.df = df.copy()
        self.metrics_column = metrics_column
        self.category_column = category_column
        self.model_type = model_type
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


    def tune_params(self,df, metrics_column):
        eps = 1e-8
        cont_max = .1
        k_max = 50
        cont_steps = 100
        k_grid = np.arange(1,k_max + 1) #neighbors
        cont_grid = np.linspace(0.005, cont_max, cont_steps) #contamination
        collector = []
        n_samples = df.shape[0]

        for contamination in tqdm.tqdm(cont_grid):
            samps = int(contamination * n_samples)
            if samps < 2:
                continue

            #init running metrics
            running_metrics = defaultdict(list)
            for k in k_grid:
                clf = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
                if (type(metrics_column) == list):
                    clf.fit_predict(df[metrics_column])
                else:
                    clf.fit_predict(df[[metrics_column]])
                X_scores = np.log(- clf.negative_outlier_factor_)
                t0 = X_scores.argsort()#[::-1]
                top_k = t0[-samps:]
                min_k = t0[:samps]

                x_out = X_scores[top_k]
                x_in = X_scores[min_k]

                mc_out = np.mean(x_out)
                mc_in = np.mean(x_in)
                vc_out = np.var(x_out)
                vc_in = np.var(x_in)
                Tck = (mc_out - mc_in)/np.sqrt((eps + ((1/samps)*(vc_out +vc_in))))

                running_metrics['tck'].append(Tck)
                running_metrics['mck_out'].append(mc_out)
                running_metrics['mck_in'].append(mc_in)
                running_metrics['vck_in'].append(vc_in)
                running_metrics['vck_out'].append(vc_out)

            largest_idx = np.array(running_metrics['tck']).argsort()[-1]
            mean_mc_out = np.mean(running_metrics['mck_out'])
            mean_mc_in = np.mean(running_metrics['mck_in'])
            mean_vc_out = np.mean(running_metrics['vck_out'])
            mean_vc_in = np.mean(running_metrics['vck_in'])

            ncpc = (mean_mc_out - mean_mc_in)/np.sqrt((eps + ((1/samps)*(mean_vc_out
                                                                         + mean_vc_in))))
            dfc = (2*samps) - 2

            if dfc <= 0:
                continue

            Z = nct(dfc, ncpc)
            K_opt = k_grid[largest_idx]
            T_opt = running_metrics['tck'][largest_idx]
            Z = Z.cdf(T_opt)
            collector.append([K_opt, T_opt, Z, contamination])

        max_cdf = 0.
        self.tuned_params = {}
        for v in collector:
            Kopt, T_opt, Z, contamination = v
            if Z > max_cdf:
                max_cdf = Z

            if max_cdf == Z:
                self.tuned_params['k'] = K_opt
                self.tuned_params['c'] = contamination
        print("\nTuned LOF Parameters : {}".format(self.tuned_params))
        return(self.tuned_params)


    def create_model(self, **model_args):
        if self.model_type == 'LODA':
            self.model = LODA(**model_args)
            return self.model
        if self.model_type == 'OneClassSVM':
            self.model = OneClassSVM(**model_args)
            return self.model
        if self.model_type == 'LocalOutlierFactor':
            res = input("Would you like to automatically tune hyperparameters? ")
            if res == 'yes':
                print("Automatically tuning parameters")
                tuned_params = self.tune_params(self.df, self.metrics_column)
                self.model = LocalOutlierFactor(n_neighbors=tuned_params['k'], contamination=tuned_params['c'])
                return self.model
            else:
                self.model = LocalOutlierFactor(**model_args)
                return self.model
        if self.model_type == 'IsolationForest':
            self.model = IsolationForest(**model_args)
            return self.model


    def predict(self, df = None):
        if df is None:
            df = self.df
            if self.ScalerClass:
                scaler = self.ScalerClass()
                scaler = scaler.fit(df[[self.metrics_column]])
                df[self.metrics_column] = scaler.transform(df[[self.metrics_column]])

        if self.model_type == 'LODA':
            score_df = df.copy()
            numeric_features = list(score_df.select_dtypes(include=['int64', 'float64']).columns)
            score_df['score'] = self.model.fit_predict(score_df[numeric_features])
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
            return score_df
        if (type(self.metrics_column) == list):
            score_df = df.copy()
            score_df['score'] = self.model.fit_predict(score_df[self.metrics_column])
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
            return score_df
        else:
            score_df = df.copy()
            score_df['score'] = self.model.fit_predict(score_df[[self.metrics_column]].values)
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
            return score_df

    def get_anomalies(self):
        if self.score_df is not None:
            return self.score_df.loc[self.score_df['anomaly'] == True]

    def graph(self, no_geo: bool = False, three_d: bool = False, **kwargs):
        if (three_d and no_geo and ((self.model_type == 'LODA') or type(self.metrics_column) == list)):
            if self.model_type == 'LODA':
                numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
                X = self.score_df[numeric_features].values
            else:
                X = self.score_df[self.metrics_column].values
            pca = PCA(n_components=3)
            components = pca.fit_transform(X)
            total_var = pca.explained_variance_ratio_.sum() * 100
            score_df = self.score_df
            score_df = pd.concat([self.score_df, pd.DataFrame(components)], axis=1)
            score_df = score_df.rename(columns={0:'PC1', 1:'PC2', 2:'PC3'})
            score_df = pd.concat([score_df, pd.DataFrame(self.model.negative_outlier_factor_)], axis=1)
            score_df = score_df.rename(columns={0:'score1'})
            score_df['score1'] = round(-1*score_df['score1'],4)
            fig = px.scatter_3d(
                score_df, x='PC1', y='PC2', z='PC3', 
                color='anomaly',size='score1',size_max=25,
                title=f'Total Explained Variance: {total_var:.2f}%',
                hover_data = [self.category_column],
                )
            return fig.show()
        
        if (no_geo and ((self.model_type == 'LODA') or type(self.metrics_column) == list)):
            if self.model_type == 'LODA':
                numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
                X = self.score_df[numeric_features].values
                pca = PCA(n_components=2)
                components = pca.fit_transform(X)
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                score_df = self.score_df
                score_df['PC1'] = components[:,0]
                score_df['PC2'] = components[:,1]
                fig = px.scatter(score_df, x='PC1', y='PC2', title='PCA Plot - Anomalies',
                                 color='anomaly', hover_data=[self.category_column],
                                 )
                for i, feature in enumerate(numeric_features):
                    fig.add_shape(
                        type='line',
                        x0=0, y0=0,
                        x1=loadings[i, 0],
                        y1=loadings[i, 1]
                            )
                    fig.add_annotation(
                            x=loadings[i, 0],
                            y=loadings[i, 1],
                            ax=0, ay=0,
                            xanchor="center",
                            yanchor="bottom",
                            text=feature,
                                )
                return fig.show()
            else:
                X = self.score_df[self.metrics_column].values
                pca = PCA(n_components=2)
                components = pca.fit_transform(X)
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                score_df = pd.concat([self.score_df, pd.DataFrame(components)], axis=1)
                score_df = score_df.rename(columns={0:'PC1', 1:'PC2'})

                fig = px.scatter(score_df, x='PC1', y='PC2', title='PCA Plot - Anomalies',
                                 color='anomaly', hover_data=[self.category_column],
                                 )

                for i, feature in enumerate(self.metrics_column):
                    fig.add_shape(
                            type='line',
                            x0=0, y0=0,
                            x1=loadings[i, 0],
                            y1=loadings[i, 1]
                            )
                    fig.add_annotation(
                            x=loadings[i, 0],
                            y=loadings[i, 1],
                            ax=0, ay=0,
                            xanchor="center",
                            yanchor="bottom",
                            text=feature,
                            )
                return fig.show()

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

        if self.model_type:
            if type(self.metrics_column) == str:
                hover = [self.metrics_column]
            else:
                hover = self.metrics_column
            hover.append(self.category_column)
            hover.append('anomaly')
            plotly_args = {
                'data_frame': self.score_df,
                'geojson': json.loads(self.score_df.to_json()),
                'featureidkey': f'properties.{self.category_column}',
                'locations': self.category_column,
                'color': 'anomaly',
                'hover_data': hover,
                'opacity': 0.5,
                'mapbox_style': 'carto-positron',
                'center': {'lat': 44, 'lon': -79},
                'zoom': 7,
                'height': 720,
                **kwargs,
            }
            return px.choropleth_mapbox(**plotly_args)
        else:
            return("Please input an appropriate model name")
