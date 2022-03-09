from typing import Union
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import tqdm

import rasterio
import rasterio.mask

from collections import defaultdict

import importlib

import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from anlearn.loda import LODA
from scipy.stats import nct


def get_population_count(vector_polygon, raster_layer):
    gtraster = rasterio.mask.mask(raster_layer, [vector_polygon], crop=True)
    pop_estimate = gtraster[0][gtraster[0] > 0].sum()
    return pop_estimate.round(2)


class CategoricalAnomalyDetector:

    def __init__(
        self,
        df: Union[pd.DataFrame, gpd.GeoDataFrame],
        metrics_column: str,
        category_column: str,
        model_type: str,
        geotif_path: str,
        ScalerClass: str,
    ):
        self.df = df.copy()
        self.metrics_column = metrics_column
        self.category_column = category_column
        self.model_type = model_type
        self.ScalerClass = ScalerClass
        self.score_df = None
        self.scaled_df = None
        self.geotif_path = geotif_path


    def scale(self,):
        """
        Uses importlib to allow users to select a variety of scaling
        options from sklearn's preprocessing module.
        !! IMPORTANT !!
        Users should scale their data prior to creating / fitting /
        predicting with their model
        """
        module = importlib.import_module('sklearn.preprocessing')
        class_ = getattr(module, self.ScalerClass)
        scaler = class_()
        numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
        self.scaled_df = self.df.copy()
        self.scaled_df[numeric_features] = scaler.fit_transform(self.df[numeric_features])
        return self.scaled_df


    def tune_params(self, df, metrics_column):
        """
        From this article: https://arxiv.org/pdf/1902.00567.pdf it's
        possible to tune the hyperparameters for a LOF model, if a
        user selects auto_tune=True during model creation, this code will run
        and the parameters will be automatically tuned and selected
        """
        eps = 1e-8
        cont_max = .1
        k_max = 50
        cont_steps = 100
        k_grid = np.arange(1, k_max + 1)  # neighbors
        cont_grid = np.linspace(0.005, cont_max, cont_steps)  # contamination
        collector = []
        n_samples = df.shape[0]

        for contamination in tqdm.tqdm(cont_grid):
            samps = int(contamination * n_samples)
            if samps < 2:
                continue

            # init running metrics
            running_metrics = defaultdict(list)
            for k in k_grid:
                clf = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
                if (type(metrics_column) == list):
                    clf.fit_predict(df[metrics_column])
                else:
                    clf.fit_predict(df[[metrics_column]])
                X_scores = np.log(- clf.negative_outlier_factor_)
                t0 = X_scores.argsort()  # [::-1]
                top_k = t0[-samps:]
                min_k = t0[:samps]

                x_out = X_scores[top_k]
                x_in = X_scores[min_k]

                mc_out = np.mean(x_out)
                mc_in = np.mean(x_in)
                vc_out = np.var(x_out)
                vc_in = np.var(x_in)
                Tck = (mc_out - mc_in)/np.sqrt((eps + ((1/samps)*(vc_out + vc_in))))

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

            ncpc = (mean_mc_out - mean_mc_in)/np.sqrt((eps + ((1/samps)*(mean_vc_out + mean_vc_in))))
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


    def raster_to_vector(self, geotif_path=None):
        """
        Given a filepath to a geotif file, will match the first layer of
        the raster tif to the vector geometries located in the geodataframe
        given by the user.
        Ex:
        https://ghsl.jrc.ec.europa.eu/download.php?ds=pop
        Using this dataset we can match the population geotif data with
        the FSA geometries in a geocohort, and generate the population for each
        FSA
        """
        fpath = geotif_path or self.geotif_path
        with rasterio.open(fpath) as src:
            self.df['pop_count'] = self.df['geometry'].apply(get_population_count, raster_layer=src)
            return self.df


    def normalize(self, field_1 = None,):
        """
        Given a field, will normalize the numerical features in a dataset by
        that feature.
        For example, using population count for an FSA dataset will divide the
        numerical features of the dataset by the population count
        """
        numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
        numeric_features.remove(field_1)
        self.df[numeric_features] = self.df[numeric_features].div(self.df[field_1], axis=0)
        self.df = self.df.fillna(0)
        self.df = self.df.replace([np.inf, -np.inf], 0)
        return self.df


    def create_model(self, auto_tune=None, **model_args):
        if self.model_type == 'LODA':
            self.model = LODA(**model_args)
        if self.model_type == 'OneClassSVM':
            self.model = OneClassSVM(**model_args)
        if self.model_type == 'LocalOutlierFactor':
            if auto_tune:
                tuned_params = self.tune_params(self.df, self.metrics_column)
                self.model = LocalOutlierFactor(n_neighbors=tuned_params['k'],
                                                contamination=tuned_params['c'])
            else:
                self.model = LocalOutlierFactor(**model_args)
        if self.model_type == 'IsolationForest':
            self.model = IsolationForest(**model_args)
        return self.model


    def predict(self, df=None):
        """
        Runs fit_predict on a users data, with their given inputs,
        and returns a df with the score (-1 for anomalous, 1 for normal data)
        Attempts to use scaled data, if users haven't scaled their data,
        will default to normal data.

        LODA models cannot be run with less than 2 metrics, therefore, when a
        user selects LODA, and only 1 metric, it will automatically select all
        of the numeric features of the dataset.
        """
        if df is None:
            df = self.df
        try:
            score_df = self.scaled_df.copy()
        except:
            score_df = self.df.copy()
        if self.model_type == 'LODA' and ((type(self.metrics_column) is list) and len(self.metrics_column) > 1):
            score_df['score'] = self.model.fit_predict(score_df[self.metrics_column])
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
        elif self.model_type == 'LODA' and type(self.metrics_column) is not list or (type(self.metrics_column) is list and len(self.metrics_column) < 2):
            numeric_features = list(score_df.select_dtypes(include=['int64', 'float64']).columns)
            score_df['score'] = self.model.fit_predict(score_df[numeric_features])
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
        elif (type(self.metrics_column) is list):
            score_df['score'] = self.model.fit_predict(score_df[self.metrics_column])
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
        else:
            score_df['score'] = self.model.fit_predict(score_df[[self.metrics_column]].values)
            score_df['anomaly'] = score_df['score'] <= -1
            self.score_df = score_df
        return self.score_df


    def get_anomalies(self):
        if self.score_df is not None:
            return self.score_df.loc[self.score_df['anomaly'] == True]


    def graph_highd(self, three_d: bool = None, **kwargs):
        """
        If users have selected more than 2 metrics, this function
        allows them to apply principal component analysis to their data,
        and graph the results.

        Allows for both 2d and 3d PCA (three_d = True), and is typically
        helpful once the metrics columns are > 3

        The graph's hover data will show the non-scaled metrics_column,
        as scaled metrics / PCA numbers aren't generally helpful to users.
        The size of each point is reflective of how anomalous that point is
        - LOF uses a score called negative_outlier_factor to score how
        anomalous a point is, while LODA uses score_samples, which need to be
        rescaled to only positive integers.
        """
        try:
            if three_d:
                if self.model_type == 'LODA' and (type(self.metrics_column) is not list):
                    numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
                    feat_vals = self.score_df[numeric_features].values
                else:
                    feat_vals = self.score_df[self.metrics_column].values
                pca = PCA(n_components=3)
                components = pca.fit_transform(feat_vals)
                total_var = pca.explained_variance_ratio_.sum() * 100
                score_df = self.score_df.reset_index(drop=True)
                comp = pd.DataFrame(components).reset_index(drop=True)
                score_df = score_df.join(pd.DataFrame(comp))
                score_df = score_df.rename(columns={0:'PC1', 1:'PC2', 2:'PC3'})
                if self.model_type == 'LocalOutlierFactor':
                    score_df = pd.concat([score_df, pd.DataFrame(self.model.negative_outlier_factor_)], axis=1)  # LOF uses nof as it's score
                    score_df = score_df.rename(columns={0:'score1'})
                    score_df['score1'] = -1*score_df['score1']
                else:
                    score_df['score1'] = self.model.score_samples(self.score_df[self.metrics_column])  # LODA uses score_samples instead
                    score_df['score1'] = (score_df['score1'] - score_df['score1'].min()) / (score_df['score1'].max() - score_df['score1'].min())
                    # score_samples can be negative to positive integers, to use them as scores we need to scale them to only positive integers
                scores = self.df.reset_index().join(score_df[['score', 'anomaly', 'PC1', 'PC2', 'PC3', 'score1']])

                fig = px.scatter_3d(
                    scores, x='PC1', y='PC2', z='PC3',
                    color='anomaly',
                    size='score1',
                    size_max=25,
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    hover_data = [self.category_column],
                    )

            else:
                if self.model_type == 'LODA' and (type(self.metrics_column) != list):
                    numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
                    feat_vals = self.score_df[numeric_features].values
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(feat_vals)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    score_df = self.score_df.reset_index()
                    score_df['PC1'] = components[:,0]
                    score_df['PC2'] = components[:,1]

                    # Append values to non-scaled df so users can see
                    # the normal values for their features graphed
                    scores = self.df.reset_index().join(score_df[['score', 'anomaly', 'PC1', 'PC2',]])
                    fig = px.scatter(scores, x='PC1', y='PC2', title='PCA Plot - Anomalies',
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

                else:
                    feat_vals = self.score_df[self.metrics_column].values
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(feat_vals)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    score_df = pd.concat([self.score_df.reset_index(), pd.DataFrame(components)], axis=1).reset_index()
                    score_df = score_df.rename(columns={0:'PC1', 1:'PC2'})

                    # Append values to non-scaled df so users can see
                    # the normal values for their features graphed
                    scores = self.df.reset_index().join(score_df[['score', 'anomaly', 'PC1', 'PC2',]])
                    fig = px.scatter(scores, x='PC1', y='PC2', title='PCA Plot - Anomalies',
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
        except:
            return("Must have either model type == LODA, or metrics_column >2")


    def graph_scatter(self, **kwargs):
        """
        Plots data in a simple scatter plot. Again, hover over data will
        show the non-scaled data, so as to be more readable.
        """
        if self.model_type or not isinstance(self.score_df, gpd.GeoDataFrame):
            # Append values to non-scaled df so users can see the normal
            # values for their features graphed
            scores = self.df.join(self.score_df[['score', 'anomaly']])
            plotly_args = {
                'data_frame': scores,
                'x': self.category_column,
                'y': self.metrics_column,
                'color': 'anomaly',
                'symbol': 'anomaly',
                **kwargs,
            }
            return px.scatter(**plotly_args)


    def graph_geo(self, **kwargs):
        """
        If a uses provides a geodataframe with applicable geometries,
        this function maps those using Folium.
        """
        if type(self.metrics_column) is str:
            hover = [self.metrics_column]
        else:
            hover = self.metrics_column.copy()
        hover.append(self.category_column)
        hover.append('anomaly')
        hover.append('pop_count')

        # Append values to non-scaled df so users can see the normal values for their features graphed
        scores = self.df.join(self.score_df[['score', 'anomaly']])
        plotly_args = {
            'data_frame': scores,
            'geojson': json.loads(scores.to_json()),
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
