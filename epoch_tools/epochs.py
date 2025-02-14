import mne

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap.umap_ as umap
import pca
import hdbscan

import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import tempfile
import gzip
import os
import contextlib
import copy

from .utils import *
import warnings

class Epochs:
    def __init__(self, epochs, non_feature_cols=[], animal_id=None, condition=None):
        """
            Initialize an Epochs object.

            Parameters:
            -----------
            epochs : mne.Epochs
                The MNE Epochs object to wrap.
            non_feature_cols : list, optional
                List of metadata columns to exclude from feature selection.
            animal_id : str, optional
                Identifier for the subject (e.g., animal ID).
            condition : str, optional
                Experimental condition associated with the epochs.
        """
        if not isinstance(epochs, mne.epochs.EpochsFIF):
            raise ValueError("The provided object must be an instance of mne.Epochs.")
    
        
        self.epochs = epochs
        self.sfreq = epochs.info['sfreq']
        self.metadata = epochs.metadata
        self.condition = condition
        self.animal_id = animal_id
        self.non_feature_cols = non_feature_cols
        self.feature_cols =  [col for col in self.metadata.columns 
                                  if col not in self.non_feature_cols]
        self.features_subset = None
        self.feats = None
        self.labels = None
        self.dims = None
        self.dims_df = None
        self.reducer = None
        self.clusterer = None
        self.reducer_params = None
        self.clusterer_params = None
    
        # Clone the MNE Epochs object to inherit its functionality
    def __getattr__(self, attr):
        """
            Delegate attribute access to the wrapped mne.Epochs object.

            Parameters:
            -----------
            attr : str
                The attribute to access.

            Returns:
            --------
            Any
                The attribute value from the wrapped mne.Epochs object.
        """
        try:
            return getattr(self.epochs, attr)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __dir__(self):
        """
            Combine the attributes and methods of Epochs and the wrapped mne.Epochs object.

            Returns:
            --------
            list
                Sorted list of all available attributes and methods.
        """
        # Get attributes/methods from Epochs
        taini_attrs = set(super().__dir__())
        # Get attributes/methods from the wrapped mne.Epochs object
        epochs_attrs = set(dir(self.epochs))
        # Combine and sort
        return sorted(taini_attrs.union(epochs_attrs))
    
        # Custom serialization methods for pickle:
    def __getstate__(self):
        """Prepare the object’s state for pickling."""
        state = self.__dict__.copy()
        # Instead of pickling the MNE epochs object, we save it to a temporary file
        # and store the file’s binary content.
        with tempfile.NamedTemporaryFile(suffix='-mne-epo.fif', delete=False) as tmp:
            tmp_name = tmp.name
        # Save using MNE’s built-in method (which includes internal compression)
        with open(os.devnull, 'w') as fnull: #mute the Overwriting existing file info message
            with contextlib.redirect_stdout(fnull):
                self.epochs.save(tmp_name, overwrite=True)
        # Read the saved file into bytes
        with open(tmp_name, 'rb') as f:
            state['epochs_bytes'] = f.read()
        # Clean up: remove the temporary file
        os.remove(tmp_name)
        # Remove the original epochs attribute from the state
        del state['epochs']
        return state

    def __setstate__(self, state):
        """Restore the object’s state from the pickled state."""
        # Extract the saved bytes and write them back to a temporary file.
        epochs_bytes = state.pop('epochs_bytes')
        with tempfile.NamedTemporaryFile(suffix='-mne-epo.fif', delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(epochs_bytes)
        # Re-load the MNE epochs object
        self.epochs = mne.read_epochs(tmp_name)
        os.remove(tmp_name)
        # Update the instance dictionary with the remaining state
        self.__dict__.update(state)
    

    def __getitem__(self, item):
        """
            Retrieve a subset of the Epochs object.

            Parameters:
            -----------
            item : int, slice, or array-like
                Index or indices to subset the epochs.

            Returns:
            --------
            Epochs
                A new Epochs object containing the subset.
        """
        subset_epochs = self.epochs[item]
        new = Epochs(
            subset_epochs,
            non_feature_cols=self.non_feature_cols,
            animal_id=self.animal_id,
            condition=self.condition,
        )
        new.metadata = new.metadata.reset_index(drop=True)
        # Also slice features/labels
        if self.feats is not None:
            new.feats = self.feats[item]
            new.feats = new.feats.reset_index(drop=True)
        if self.labels is not None:
            new.labels = self.labels[item]
            new.labels = new.labels.reset_index(drop=True)
        return new
    
    # Feature selection methods

    def create_feature_subset(self, features=None, ch_names=None):
        """
            Subset features of interest. If features is not provided,
                then a widget appears to select features

            Parameters:
            -----------
            features : list, optional
                List of feature names to include in the subset.
            ch_names : list or dict, optional
                Channel names to filter features by.
        """
        
        if features:
            assert isinstance(features, list)
            self.features_subset = features
        else:
            if ch_names:
                cols = [col for col in self.feature_cols if any(ch_name in col for ch_name in ch_names)]
            else:
                cols = self.feature_cols
            column_selector = widgets.SelectMultiple(
                        options=cols,
                        description="Feature Names",
                        disabled=False,
                        layout=widgets.Layout(width='80%', height='200px')
                    )
            button = widgets.Button(description="Select Feature Subset")
            display(column_selector, button)
            def callback(b):
                self.features_subset = list(column_selector.value)
                print("Feature Subset Set")
                print(column_selector.value)
            button.on_click(callback)
            # column_selector.close()
            # button.close()

    def get_features_df(self):
        """
            Retrieve the metadata features as a DataFrame.

            Returns:
            --------
            pd.DataFrame
                The metadata features.
        """
        return self.metadata[self.feature_cols]
    
    def get_features(self, scaler=None, dropna=False, ch_names = None, as_array=False):
        """
            Extract features from the metadata with optional scaling.

            Parameters:
            -----------
            scaler : str or None, optional
                Scaling method ('minmax', 'standard', or None).
            dropna : bool, optional
                Whether to drop NaN values.
            ch_names : list or dict, optional
                Filter features by specific channel names.
            as_array : bool, optional
                Whether to return features as a NumPy array.

            Returns:
            --------
            pd.DataFrame or np.ndarray
                The processed feature data.
        """
        
        feats = self.metadata.copy()

        if dropna:
            feats = feats.dropna()
            
        # Selecting features subset
        if self.features_subset:
            feats = feats[self.features_subset]
        else:
            feats = feats[self.feature_cols]

        if ch_names:
            if isinstance(ch_names, dict):
                feats = feats[[col for col in feats.columns if any(ch_name in col for ch_name in ch_names)]]
                #rename 
                for old_name, new_name in ch_names.items():
                    feats.columns = feats.columns.str.replace(old_name, new_name, regex=False)
            elif isinstance(ch_names, list):
                feats = feats[[col for col in feats.columns if any(ch_name in col for ch_name in ch_names)]]
            else:
                raise ValueError("""ch_names must be a dictionary [channel_name]:[channel_rename]
                                 or a list [channel_names].""")
            
        if scaler == 'minmax':
            scaler = MinMaxScaler()
            feats = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns, index=feats.index)
        elif scaler == 'standard':
            scaler = StandardScaler()
            feats = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns, index=feats.index)
        elif scaler is None:
            pass  # No scaling applied
        else:
            raise ValueError("Invalid scaler. Provide a callable scaler or one of ['minmax', 'standard', None].")
        
        # Set current features
        self.feats = feats

        if as_array:
            return feats.to_numpy()
        else:
            return feats
    
    # Epoch analysis
    
    def plot_epoch(self, idx=None, channels='all'):
        """
            Plot the signal of a specific epoch.

            Parameters:
            -----------
            idx : int, optional
                Index of the epoch to plot. Defaults to a random selection.
            channels : list or 'all', optional
                Channels to plot. Defaults to all channels.
        """
        idx = np.random.choice(self.epochs.metadata.index) if idx is None else idx

        if channels == 'all':
            n_channels = len(self.epochs.ch_names)
            channels = self.epochs.ch_names
        else:
            assert isinstance(channels, list)
            n_channels = len(channels)
        
        fig, ax = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
        fig.suptitle(f"Epoch {idx} - ID: {self.animal_id} - Condition: {self.condition}", y=1.000005, fontsize=14)
        ax = [ax] if n_channels == 1 else ax
        for i, ch in enumerate(channels):
            ax[i].plot(self.epochs.get_data(picks=ch)[idx, :, :].T, color='black', alpha=0.85)
            ax[i].set_title(ch)

        plt.tight_layout()

    def compute_psd_(self, channels='all', fmin=0, fmax=100, epoch_idx = None,**kwargs):
        """
            Compute the power spectral density (PSD) of the epochs.

            Parameters:
            -----------
            channels : list or 'all', optional
                Channels to compute PSD for.
            fmin : float, optional
                Minimum frequency (Hz) to include.
            fmax : float, optional
                Maximum frequency (Hz) to include.
            epoch_idx : int, optional
                Specific epoch index to compute PSD for.

            Returns:
            --------
            tuple
                PSD values and corresponding frequencies.
        """
        if epoch_idx is None:
            psd, freq = mne.time_frequency.psd_array_multitaper(
                self.epochs.get_data(picks=channels), sfreq=self.sfreq, fmin=fmin, fmax=fmax, **kwargs
            )
        else:
            psd, freq = mne.time_frequency.psd_array_multitaper(
                self.epochs.get_data(picks=channels)[epoch_idx], sfreq=self.sfreq, fmin=fmin, fmax=fmax, **kwargs
            )
        return psd, freq
    
    def plot_psd_(self, channels='all', fmin=0, fmax=100, 
                  log=True, norm=False, err_method='sd', epoch_idx = None, **kwargs):
        """
            Plot the power spectral density (PSD) of the epochs.

            Parameters:
            -----------
            channels : list or 'all', optional
                Channels to plot PSD for.
            fmin : float, optional
                Minimum frequency (Hz) to include.
            fmax : float, optional
                Maximum frequency (Hz) to include.
            log : bool, optional
                Whether to use a logarithmic scale.
            norm : bool, optional
                Whether to normalize PSD values.
            err_method : str, optional
                Error computation method ('sd', 'sem', 'ci', or None).
            epoch_idx : int, optional
                Specific epoch index to plot PSD for.
        """
        
        freq_bands = {
            r'$\delta$': (1, 4),  # Delta
            r'$\theta$': (4, 8),  # Theta
            r'$\alpha$': (8, 13),  # Alpha
            r'$\beta$': (13, 30),  # Beta
            r'$\gamma$': (30, 100)  # Gamma
        }


        if channels == 'all':
            n_channels = len(self.epochs.ch_names)
            channels = self.epochs.ch_names
        else:
            assert isinstance(channels, list)
            n_channels = len(channels)

        psd, freq = self.compute_psd_(channels, fmin, fmax, epoch_idx=epoch_idx, **kwargs)

        if norm:
            psd /= np.sum(psd)

        if err_method == 'sd':
            err = np.std(psd, axis=0)
        elif err_method == 'sem':
            err = np.std(psd, axis=0) / np.sqrt(psd.shape[0])
        elif err_method == "ci":
            err = np.std(psd, axis=0)
            err = 1.96 * err / np.sqrt(psd.shape[0])
        elif err_method == None:
            err = None
        else:
            raise ValueError("Invalid error method. Provide one of ['sd', 'sem', 'ci', 'none'].")


        nrows, ncols = row_col_layout(n_channels)
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))
        
        fig.suptitle(f"PSD - ID: {self.animal_id} - Condition: {self.condition}\nEpoch {epoch_idx}",
                      y=1.000005, fontsize=14)
        ax = [ax] if n_channels == 1 else ax.ravel()
        
        for i, ch in enumerate(channels):
            if epoch_idx is None:
                _psd = np.mean(psd[:, i, :], axis=0)
            else:
                _psd = psd[i , :]
                err = None

            ax[i].plot(freq, _psd, color='black', alpha=0.85)
            ax[i].set_title(ch)
            ax[i].set_xlabel('Frequency (Hz)')
            ax[i].set_ylabel('Power Spectral Density (mV/Hz)')
            if log:
                ax[i].set_yscale('log')
            if err is not None:
                ax[i].fill_between(freq, _psd - err[i, :],  _psd + err[i, :], color='black', alpha=0.2)
            for band, (fmin_, fmax_) in freq_bands.items():
                # vertical lines for frequency bands
                ax[i].axvline(fmin_, color='black', linestyle='--', alpha=0.1)
                ax[i].axvline(fmax_, color='black', linestyle='--', alpha=0.1)
                # text for frequency bands
                # ax[i].text((fmin_ + fmax_) / 2, ax[i].get_ylim()[1] * 1.01, band, 
                #            horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')
        plt.tight_layout()

    # Clustering methods
    def cluster_data(
        self,
        reducer=None,            # one of {None, 'umap', 'pca', 't-sne'}
        clusterer='kmeans',      # one of {'kmeans', 'hdbscan'}
        reducer_params=None,     # dict of parameters for the reducer
        clusterer_params=None    # dict of parameters for the clusterer
    ):
        """
            Clusters data based on an optional dimensionality reducer and a chosen clustering algorithm.

            TODO: Make it also a standalone method

            Parameters
            ----------
            data : pd.DataFrame or np.ndarray
                Already scaled feature data. Shape = [n_samples, n_features].
            reducer : str or None, optional
                Which dimensionality reducer to use ('umap', 'pca', 't-sne' or None).
                If None, no dimensionality reduction is applied.
            clusterer : str, optional
                Which clustering algorithm to use ('kmeans' or 'hdbscan').
            reducer_params : dict, optional
                Dictionary of hyperparameters for the reducer.
                Example for UMAP: {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2, "random_state": 42}
            clusterer_params : dict, optional
                Dictionary of hyperparameters for the clusterer.
                Example for KMeans: {"n_clusters": 5, "random_state": 42}

            Returns
            -------
            labels : np.ndarray
                Array of cluster assignments for each row in `data`.
            """

        data = self.feats.values  # convert to NumPy

        # Set defaults if dictionaries are None TODO: Add default values
        if reducer_params is None:
            reducer_params = {}
        if clusterer_params is None:
            clusterer_params = {}
        
        self.reducer = str(reducer)
        self.clusterer = str(clusterer)
        self.reducer_params = copy.deepcopy(reducer_params)
        self.clusterer_params = copy.deepcopy(clusterer_params)

        # 1. Dimensionality Reduction (if applied)
        if reducer == 'umap':
            reducer_model = umap.UMAP(**reducer_params)
            data_reduced = reducer_model.fit_transform(data)

        elif reducer == 'pca':
            reducer_model = PCA(**reducer_params)
            data_reduced = reducer_model.fit_transform(data)

        elif reducer == 't-sne':
            reducer_model = TSNE(**reducer_params)
            data_reduced = reducer_model.fit_transform(data)

        else:
            data_reduced = data
        self.dims = data_reduced.shape[1]
        self.dims_df = pd.DataFrame(data_reduced, columns=[f'dim{i+1}' for i in range(self.dims)])

        # 2. Clustering
        if clusterer == 'kmeans':
            clusterer_model = KMeans(**clusterer_params)
            labels = clusterer_model.fit_predict(data_reduced)

        elif clusterer == 'hdbscan':
            clusterer_model = hdbscan.HDBSCAN(**clusterer_params)
            labels = clusterer_model.fit_predict(data_reduced)

        else:
            raise ValueError("Unsupported clustering method. Choose from 'kmeans' or 'hdbscan'.")

        self.labels = labels

    def plot_dim_reduction(self, dims=(1,2), ax=None, plot3d=False, plot_outliers=True,
                  plot_labels=True, palette='tab10', edgecolor='black', s=10, alpha=0.5, 
                  figsize3d = (10, 10), **kwargs):
        """
            Plot the dimensionaliry reduction results from self.dims_df
        """
        if self.dims_df is None:
            raise ValueError("No dimensionality reduction results found. Run cluster_data() first.")
        
        plot_df = self.dims_df.copy()
        if self.labels is not None:
            plot_df['label'] = self.labels
        
        if not plot_outliers:
            plot_df = plot_df[plot_df['label'] != -1]

        if not all(f'dim{dim}' in self.dims_df.columns for dim in dims):
            raise ValueError(f"One or more specified dimensions {dims} do not exist in the dimensionality reduction results. Current availabel dimensions {self.dims_df.columns}.")

        dim1, dim2 = f'dim{dims[0]}', f'dim{dims[1]}'

        if plot3d and len(dims) >= 3:
            dim3 = f'dim{dims[2]}'
            fig = plt.figure(figsize=figsize3d)
            ax = fig.add_subplot(111, projection='3d')
            if self.labels is not None and plot_labels:
                ax.scatter(plot_df[dim1], plot_df[dim2], plot_df[dim3], c=plot_df['label'], cmap=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            else:
                ax.scatter(plot_df[dim1], plot_df[dim2], plot_df[dim3], cmap=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
                ax.set_xlabel(dim1)
                ax.set_ylabel(dim2)
                ax.set_zlabel(dim3)
            return ax
        else:
            if ax is None:
                fig, ax = plt.subplots()
            if self.labels is not None and plot_labels:
                sns.scatterplot(data=plot_df, x=dim1, y=dim2, hue='label', palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Label')
            else:
                sns.scatterplot(data=plot_df, x=dim1, y=dim2, ax=ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
                ax.set_xlabel(dim1)
                ax.set_ylabel(dim2)
            return ax


    # Feature vizualization methods
    def plot_simple_pca(self,  n_components=2, title='', **kwargs):
        """
            Perform and plot a simple PCA on features.

            Parameters:
            -----------
            n_components : int, optional
                Number of principal components to use.
            title : str, optional
                Title of the plot.
        """

        pca = PCA(n_components=n_components, **kwargs)
        pca.fit(self.feats)
        X_pca = pca.transform(self.feats)

        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['label'] = self.labels
        hue = 'label' if self.labels is not None else None

        # Plot the PCA results
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', 
                        palette='viridis', ax=ax, legend=False, edgecolor='black')
        ax.set_title(title)

        if self.labels is not None:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Label')
        plt.show()

    def plot_pca(self, n_components, **kwargs):
        """
            Perform and visualize PCA with feature importance.

            Parameters:
            -----------
            n_components : int
                Number of principal components to compute.
        """

        model = pca.pca(n_components=n_components, **kwargs)
        results = model.fit_transform(X=self.feats, row_labels=self.labels)

        fig, ax = plt.subplots(ncols=2, figsize=(15, 10))

        model.scatter(s=8, ax=ax[0], cmap='turbo', edgecolor=None)
        
        sns.barplot(data=results['topfeat'].sort_values(by='PC', ascending=True), y='feature', x='loading', hue='PC' ,dodge=False, ax=ax[1])
        ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()


    def plot_tsne(self, n_components,reducer_params={}, ax=None, plot3d=False, plot_outliers=True,
                  plot_labels=True, palette='tab10', edgecolor='black', s=10, alpha=0.5, 
                  figsize3d = (10, 10), **kwargs):
        """
            Perform and plot t-SNE dimensionality reduction.

            Parameters:
            -----------
            n_components : int
                Number of components for t-SNE.
            ax : matplotlib.axes, optional
                Axes object to plot on.
            3d_plot : bool, optional
                Whether to create a 3D scatter plot.
        """
        if 'n_components' in reducer_params:
            reducer_params.pop('n_components')
        reducer = TSNE(n_components=n_components, **reducer_params)
        embedding = reducer.fit_transform(self.feats)
        plot_df = pd.DataFrame(embedding, columns=[f't-SNE {i+1}' for i in range(embedding.shape[1])])

        if self.labels is not None:
            plot_df['label'] = self.labels
        
        if not plot_outliers:
            plot_df = plot_df[plot_df['label'] != -1]

        if plot3d and n_components >= 3:
            fig = plt.figure(figsize=figsize3d)
            ax = fig.add_subplot(111, projection='3d')
            if self.labels is not None and plot_labels:
                ax.scatter(plot_df['t-SNE 1'], plot_df['t-SNE 2'], plot_df['t-SNE 3'], c=plot_df['label'], cmap=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            else:
                ax.scatter(plot_df['t-SNE 1'], plot_df['t-SNE 2'], plot_df['t-SNE 3'], cmap=palette, edgecolor=edgecolor, s=s, alpha=alpha,**kwargs)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_zlabel('UMt-SNEAP 3')
            return ax
        else:
            if ax is None:
                fig, ax = plt.subplots()
            if self.labels is not None and plot_labels:
                fig = sns.scatterplot(data=plot_df, x='t-SNE 1', y='t-SNE 2', hue='label', palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Label')
            else:
                sns.scatterplot(data=plot_df, x='t-SNE 1', y='t-SNE 2', ax=ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            return ax  


    def plot_umap(self, n_components,reducer_params={}, ax=None, plot3d=False, plot_outliers=True,
                  plot_labels=True, palette='tab10', edgecolor='black', s=10, alpha=0.5, 
                  figsize3d = (10, 10), **kwargs):
        """
            Perform and plot UMAP dimensionality reduction.

            Parameters:
            -----------
            n_components : int
                Number of components for UMAP.
            ax : matplotlib.axes, optional
                Axes object to plot on.
            3d_plot : bool, optional
                Whether to create a 3D scatter plot.
        """
        if 'n_components' in reducer_params:
            reducer_params.pop('n_components')
        reducer = umap.UMAP(n_components=n_components, **reducer_params)

        embedding = reducer.fit_transform(self.feats)

        plot_df = pd.DataFrame(embedding, columns=[f'UMAP {i+1}' for i in range(embedding.shape[1])])

        if self.labels is not None:
            plot_df['label'] = self.labels
        
        if not plot_outliers:
            plot_df = plot_df[plot_df['label'] != -1]

        if plot3d and n_components >= 3:
            fig = plt.figure(figsize=figsize3d)
            ax = fig.add_subplot(111, projection='3d')
            if self.labels is not None and plot_labels:
                ax.scatter(plot_df['UMAP 1'], plot_df['UMAP 2'], plot_df['UMAP 3'], c=plot_df['label'], cmap=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            else:
                ax.scatter(plot_df['UMAP 1'], plot_df['UMAP 2'], plot_df['UMAP 3'], cmap=palette, edgecolor=edgecolor, s=s, alpha=alpha,**kwargs)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            return ax
        else:
            if ax is None:
                fig, ax = plt.subplots()
            if self.labels is not None and plot_labels:
                fig = sns.scatterplot(data=plot_df, x='UMAP 1', y='UMAP 2', hue='label', palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Label')
            else:
                sns.scatterplot(data=plot_df, x='UMAP 1', y='UMAP 2', ax=ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            return ax  


    def plot_feature_correlation(self, threshold=0.8, ax=None, 
                                 cmap='coolwarm', vmin=-1, vmax=1, annot=False, fmt='.2f', **kwargs):
        """
            Plot the correlation matrix of extracted features and mark correlations above a threshold.

            Parameters:
            -----------
            threshold : float, optional
                Threshold above which to mark the correlations.
            ax : matplotlib.axes, optional
                Axes object to plot on.
        """
        corr = self.feats.corr()
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 12))
        
        sns.heatmap(corr, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=annot, fmt=fmt, **kwargs)
        ax.set_title('Feature Correlation Matrix')

        # Mark correlations above the threshold
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold:
                    ax.text(j + 0.5, i + 0.5, '*', color='black', ha='center', va='center')

        return ax
    
    def plot_hierarchy(self):
        pass

    # I/O methods
    @staticmethod
    def _parse_base_name(fname):
        """
            Validate the provided filename and extract the base name without extension.

            Parameters:
            fname (str): The provided filename.

            Returns:
            str: The base filename without extension.
        """
        if fname.endswith('-mne-epo.fif'):
            return fname.replace('-mne-epo.fif', '')
        elif fname.endswith('-taini-epo.pkl'):
            return fname.replace('-taini-epo.pkl', '')
        elif '.' in fname:
            raise ValueError("Invalid file extension. Expected '-mne-epo.fif' or '-taini-epo.pkl'.")
        return fname

    def save_epochs(self, fname, overwrite=False):
        """
            DEPRECATED:
            Save the Epochs object.

            Parameters:
            fname (str): The base filename.
            overwrite (bool): Whether to overwrite existing files.
        """
        # depraecated warning
        warnings.warn("save_epochs() method is deprecated and will be removed in the next version. Use save_gz() instead.", DeprecationWarning)

        # Validate and extract base filename
        base_fname = self._parse_base_name(fname)
        mne_fname = f"{base_fname}-mne-epo.fif"
        pickle_fname = f"{base_fname}-taini-epo.pkl"

        # Check for overwrite
        if not overwrite and (os.path.exists(mne_fname) or os.path.exists(pickle_fname)):
            raise FileExistsError(
                f"Files '{mne_fname}' or '{pickle_fname}' already exist. Use overwrite=True to overwrite."
            )

        # Save MNE Epochs
        self.epochs.save(mne_fname, overwrite=overwrite)

        # Save additional attributes to pickle
        attributes_to_save = {key: value for key, value in self.__dict__.items() if key != "epochs"}
        with open(pickle_fname, "wb") as f:
            pickle.dump(attributes_to_save, f)

    @classmethod
    def load_epochs(cls, fname):
        """
            Load a Epochs object.

            Parameters:
            fname (str): The base filename.

            Returns:
            Epochs: The loaded Epochs object.
        """
        warnings.warn("load_epochs() method is deprecated and will be removed in the next version. Use load_gz() instead.", DeprecationWarning)
        # Validate and extract base filename
        base_fname = cls._parse_base_name(fname)
        mne_fname = f"{base_fname}-mne-epo.fif"
        pickle_fname = f"{base_fname}-taini-epo.pkl"

        # Check if both files exist
        if not os.path.exists(mne_fname):
            raise FileNotFoundError(f"File '{mne_fname}' not found.")
        if not os.path.exists(pickle_fname):
            raise FileNotFoundError(f"File '{pickle_fname}' not found.")

        # Load MNE Epochs
        epochs = mne.read_epochs(mne_fname)

        # Load additional attributes
        with open(pickle_fname, "rb") as f:
            attributes = pickle.load(f)

        # Reconstruct the object
        obj = cls(epochs)
        obj.__dict__.update(attributes)
        return obj
    
    def save_gz(self, fname, overwrite=False):
        """
        Save the entire object (including MNE Epochs) into a single gzip-compressed file.
        
        Parameters:
            fname (str): Filename for the saved file (e.g., 'my_epochs.gz').
            overwrite (bool): If False and file exists, raises an error.
        """
        # Check filname extension
        if not fname.endswith('.gz'):
            raise ValueError("Filename must end with '.gz'.")
        if not overwrite and os.path.exists(fname):
            raise FileExistsError(f"File '{fname}' already exists.")
        with gzip.open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_gz(cls, fname):
        """
        Load the object from a gzip-compressed file.
        
        Parameters:
            fname (str): The filename to load (e.g., 'my_epochs.gz').
            
        Returns:
            Epochs: The loaded Epochs object.
        """
        with gzip.open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj



