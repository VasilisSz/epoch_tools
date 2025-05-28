from .utils import row_col_layout, compute_err

import mne
from mne.epochs import BaseEpochs

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap.umap_ as umap
import pca
import hdbscan
from tqdm import tqdm

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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                        module="sklearn.utils.deprecation")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


class Epochs:
    def __init__(self, epochs, non_feature_cols=[], 
                 animal_id=None, condition=None):
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
        if not isinstance(epochs, BaseEpochs):
            raise ValueError(
                "The provided object must be an MNE Epochs instance "
                f"(got {type(epochs)})."
            )
        self.epochs = epochs
        self.sfreq = epochs.info['sfreq']
        self.metadata = epochs.metadata
        self.condition = condition
        self.animal_id = animal_id
        self.non_feature_cols = non_feature_cols
        self.feature_cols = [
            col for col in self.metadata.columns if
            col not in self.non_feature_cols
        ]
        self.features_subset = None
        self.feats = None
        self.labels = None
        self.dims = None
        self.dims_df = None
        self.reducer = None
        self.clusterer = None
        self.reducer_params = None
        self.clusterer_params = None

        self.psd_results = {}  # Cache for PSD results

        self.extra_info = {} # Placeholder for additional information

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
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
                )

    def __dir__(self):
        """
            Combine the attributes and methods of Epochs 
            and the wrapped mne.Epochs object.

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
        # We save it to a temporary file
        # and store the file’s binary content.
        with tempfile.NamedTemporaryFile(suffix='-mne-epo.fif',
                                         delete=False) as tmp:
            tmp_name = tmp.name
        # Save using MNE built-in method (which includes internal compression)
        with open(os.devnull, 'w') as fnull:  # mute the Overwriting warning
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
        with tempfile.NamedTemporaryFile(suffix='-mne-epo.fif',
                                         delete=False) as tmp:
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
        if isinstance(item, pd.Index):
            item = item.tolist()

        subset_epochs = self.epochs[item]
        new = Epochs(
            subset_epochs,
            non_feature_cols=self.non_feature_cols,
            animal_id=self.animal_id,
            condition=self.condition,
        )
        if isinstance(item, (list, np.ndarray)):
            new.metadata = self.metadata.iloc[item]
            # Also slice features/labels
            try:
                if self.feats is not None:
                    new.feats = self.feats.iloc[new.metadata.index]
                    new.feats = new.feats.reset_index(drop=True)
                if self.labels is not None:
                    new.labels = self.labels[new.metadata.index]
            except IndexError:
                print("IndexError: The index of the features/labels does not match the metadata index.")
                print("Resetting features/labels to None.")
                new.feats = None
                new.labels = None
        else:
            new.metadata = self.metadata[item]
            # Also slice features/labels
            try:
                if self.feats is not None:
                    new.feats = self.feats.iloc[new.metadata.index]
                    new.feats = new.feats.reset_index(drop=True)
                if self.labels is not None:
                    new.labels = self.labels[new.metadata.index]
            except IndexError:
                print("IndexError: The index of the features/labels does not match the metadata index.")
                print("Resetting features/labels to None.")
                new.feats = None
                new.labels = None
        
        return new
        

    def __deepcopy__(self, memo):
        # Create a new instance with a copied MNE Epochs
        new_instance = self.__class__(
            self.epochs.copy(),
            non_feature_cols=copy.deepcopy(self.non_feature_cols, memo),
            animal_id=self.animal_id,
            condition=self.condition,
        )
        new_instance.metadata = (
            self.metadata.copy() if self.metadata is not None else None
        )
        new_instance.features_subset = (
            copy.deepcopy(self.features_subset, memo)
        )
        new_instance.feats = (
            self.feats.copy() if self.feats is not None else None
        )
        new_instance.labels = (
            self.labels.copy() if self.labels is not None else None
        )
        new_instance.dims = self.dims
        new_instance.dims_df = (
            self.dims_df.copy() if self.dims_df is not None else None
        )
        new_instance.reducer = copy.deepcopy(self.reducer, memo)
        new_instance.clusterer = copy.deepcopy(self.clusterer, memo)
        new_instance.reducer_params = copy.deepcopy(self.reducer_params, memo)
        new_instance.clusterer_params = (
            copy.deepcopy(self.clusterer_params, memo)
        )
        return new_instance
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
                cols = [col for col in self.feature_cols if
                        any(ch_name in col for ch_name in ch_names)]
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

    def get_features(self, cols=None, scaler='standard',
                     ch_names=None, as_array=False):
        """
            Extract features from the metadata with optional scaling.

            Parameters:
            -----------
            cols: list, optional
                List of feature names to include. Will override features_subset.
            scaler : str or None, optional
                Scaling method ('minmax', 'standard', or None).
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

        # Selecting features subset
        if cols:
            assert isinstance(cols, list)
            feats = feats[cols]
            if self.features_subset:
                print("Warning: features_subset is not used when cols are provided.")
        elif self.features_subset:
            feats = feats[self.features_subset]
        else:
            feats = feats[self.feature_cols]

        if ch_names:
            if isinstance(ch_names, dict):
                feats = feats[[col for col in feats.columns if
                               any(ch_name in col for ch_name in ch_names)]]
                # rename
                for old_name, new_name in ch_names.items():
                    feats.columns = feats.columns.str.replace(
                        old_name, new_name, regex=False)
            elif isinstance(ch_names, list):
                feats = feats[[col for col in feats.columns if
                               any(ch_name in col for ch_name in ch_names)]]
            else:
                raise ValueError("ch_names must be a dictionary" +
                                 "[channel_name]:[channel_rename]" +
                                 "or a list [channel_names].")

        if scaler == 'minmax':
            scaler = MinMaxScaler()
            feats = pd.DataFrame(scaler.fit_transform(feats),
                                 columns=feats.columns, index=feats.index)
        elif scaler == 'standard':
            scaler = StandardScaler()
            feats = pd.DataFrame(scaler.fit_transform(feats),
                                 columns=feats.columns, index=feats.index)
        elif scaler is None:
            pass  # No scaling applied
        else:
            raise ValueError("Invalid scaler. Provide a callable" +
                             "scaler or one of ['minmax', 'standard', None].")

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
        if idx is None:
            idx = np.random.choice(self.epochs.metadata.index)

        if channels == 'all':
            n_channels = len(self.epochs.ch_names)
            channels = self.epochs.ch_names
        else:
            assert isinstance(channels, list)
            n_channels = len(channels)

        fig, ax = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
        fig.suptitle(
            f"Epoch {idx}-ID: {self.animal_id}-Condition: {self.condition}",
            y=1.000005,
            fontsize=14
        )
        ax = [ax] if n_channels == 1 else ax
        for i, ch in enumerate(channels):
            ch_data = self.epochs[self.metadata.index==idx].get_data(picks=ch)[0][0]
            ax[i].plot(ch_data,
                       color='black', alpha=0.85)
            ax[i].set_title(ch)

        plt.tight_layout()

    def _drop_bad_channels(self, psd, channels, bad_channels):
        """
        Replace PSD values coming from bad channels with NaN.

        Parameters
        ----------
        psd : ndarray, shape (n_epochs, n_channels, n_freqs)
        channels : list[str]
            Channel names used when `compute_psd_` was called.
        bad_channels : dict[str, list[str]]
            Mapping *animal_id → list(channel_names)*

        Returns
        -------
        ndarray
            A copy of *psd* with bad-channel slices set to NaN so that
            subsequent `np.nanmean()` / `np.nanstd()` ignore them.
        """
        if not bad_channels or "animal_id" not in self.metadata.columns:
            print("Warning: No bad channels provided or 'animal_id' not in metadata.")
            return psd  # nothing to do
        # Make bad_channels keys strings
        bad_channels = {str(k): v for k, v in bad_channels.items()}
        psd = psd.copy()
        ch_map = {ch: i for i, ch in enumerate(channels)}

        # Print how many nans in psd
        print(f"Initial NaN count in PSD: {np.isnan(psd).sum()}")

        for row, animal in enumerate(self.metadata["animal_id"].to_numpy()):
            # match types of animal_id and animal in bad_channels
            animal = str(animal)
            bad = bad_channels.get(animal, [])
            idx = [ch_map[ch] for ch in bad if ch in ch_map]
            if idx:
                psd[row, idx, :] = np.nan

        # Print how many nans in psd after dropping bad channels
        print(f"NaN count in PSD after dropping bad channels: {np.isnan(psd).sum()}")
        return psd

    def compute_psd_(self, channels='all', fmin=0, fmax=100,
                     epoch_idx=None, method='multitaper', 
                     n_fft=None, n_per_seg=None,
                     **kwargs):
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
            data = self.epochs.get_data(picks=channels)
        else:
            data = self.epochs[self.metadata.index==epoch_idx].get_data(picks=channels)[0]

        if method == 'multitaper':
            psd, freq = mne.time_frequency.psd_array_multitaper(
                data, sfreq=self.sfreq,
                fmin=fmin, fmax=fmax, **kwargs
            )
        elif method == 'welch':
            # 1 Hz resolution by default
            n_fft = int(self.sfreq) if n_fft is None else n_fft
            n_per_seg = int(self.sfreq) if n_per_seg is None else n_per_seg

            psd, freq = mne.time_frequency.psd_array_welch(
                data, sfreq=self.sfreq,
                fmin=fmin, fmax=fmax, 
                n_fft=n_fft, n_per_seg=n_per_seg,
                **kwargs)
        
        # psd: shape (n_epochs, n_channels, n_freqs)
        # freq: shape (n_freqs,)
        return psd, freq

    def plot_psd_(self, channels='all', fmin=0, fmax=100,
                  log=True, norm=False, err_method='sd',
                  epoch_idx=None, **kwargs):
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

        psd, freq = self.compute_psd_(channels, fmin, fmax,
                                      epoch_idx=epoch_idx, **kwargs)

        if norm:
            psd /= np.sum(psd)

        err = compute_err(psd, err_method)

        nrows, ncols = row_col_layout(n_channels)
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, 
                               figsize=(4*ncols, 4*nrows))

        fig.suptitle(
            f"ID:{self.animal_id}-Condition:{self.condition}-Epoch{epoch_idx}",
            y=1.000005,
            fontsize=14
        )
        ax = [ax] if n_channels == 1 else ax.ravel()

        for i, ch in enumerate(channels):
            if epoch_idx is None:
                _psd = np.mean(psd[:, i, :], axis=0)
            else:
                _psd = psd[i, :]
                err = None

            ax[i].plot(freq, _psd, color='black', alpha=0.85)
            ax[i].set_title(ch)
            ax[i].set_xlabel('Frequency (Hz)')
            ax[i].set_ylabel('Power Spectral Density (mV/Hz)')
            if log:
                ax[i].set_yscale('log')
            if err is not None:
                ax[i].fill_between(freq, _psd - err[i, :],  _psd + err[i, :],
                                   color='black', alpha=0.2)
            for band, (fmin_, fmax_) in freq_bands.items():
                # vertical lines for frequency bands
                ax[i].axvline(fmin_, color='black', linestyle='--', alpha=0.1)
                ax[i].axvline(fmax_, color='black', linestyle='--', alpha=0.1)
        plt.tight_layout()

    def compare_psd(
        self,
        hue,
        *,
        channels="all",
        method="welch",
        avg_level="subject",          # {"all", "subject"}
        plot_individual_points=False,
        bad_channels=None, 
        err_method="sem",          # {"sd","sem","ci",None}
        palette="tab10",
        **kwargs,
    ):
        """
        Compare PSDs between groups defined by *hue* (one or several metadata
        columns) and plot the group-average spectra.

        Parameters
        ----------
        hue : str | list[str]
            Column(s) in *self.metadata* that define the groups.
        channels : list[str] | "all"
            Channel(s) forwarded to `compute_psd_`.
        method : {"multitaper","welch"}
            PSD estimation method (forwarded).
        avg_level : {"all","subject"}
            - ``"all"``   - average *all* epochs belonging to a group.
            - ``"subject"`` - first average within each subject
            (metadata column ``'animal_id'`` **must exist**),
            then average those subject means.
        err_method : {"sd","sem","ci",None}
            What to show as shaded error; *None* disables shading.
        palette : str | sequence
            Matplotlib / seaborn palette.
        **kwargs
            Any other keyword arguments are passed straight to `compute_psd_`.

        """

        hue_cols = [hue] if isinstance(hue, str) else list(hue)
        missing  = [c for c in hue_cols if c not in self.metadata.columns]
        if missing:
            raise ValueError(f"Hue column(s) not in metadata: {missing}")

        if avg_level not in {"all", "subject"}:
            raise ValueError("avg_level must be 'all' or 'subject'.")
 
        if channels == "all":
            ch_names = self.epochs.ch_names
        else:
            ch_names = list(channels)  # ensure list
        n_channels = len(ch_names)

        # compute PSD (n_epochs × n_channels × n_freqs)
        psd, freq = self.compute_psd_(channels=ch_names, method=method, **kwargs)
        psd       = self._drop_bad_channels(psd, ch_names, bad_channels)

        # helper DataFrame that maps every epoch to a group + subject
        helper = self.metadata[hue_cols].copy().reset_index(drop=True)
        if avg_level == "subject":
            if "animal_id" not in self.metadata.columns:
                raise ValueError("avg_level='subject' needs 'animal_id' in metadata.")
            helper["__subject__"] = self.metadata["animal_id"].values

        group_labels   = helper[hue_cols].agg(tuple, axis=1)
        unique_groups  = sorted(group_labels.unique())
        colours        = sns.color_palette(palette, n_colors=len(unique_groups))

        # containers for results
        mean_dict, err_dict, indiv_dict = {}, {}, {}

        # iterate over groups
        for grp in unique_groups:
            idx      = group_labels[group_labels == grp].index.to_numpy()
            grp_psd  = psd[idx]                               # epochs in this grp

            if avg_level == "subject":
                subj_ids = helper.loc[idx, "__subject__"]
                subj_means = []
                for subj in subj_ids.unique():
                    mask = (subj_ids == subj).to_numpy()
                    subj_means.append(np.nanmean(grp_psd[mask], axis=0))  # ch × f
                subj_means = np.stack(subj_means)                         # subj × ch × f
                group_mean = np.nanmean(subj_means, axis=0)               # ch × f
                err        = compute_err(subj_means, err_method)
                indiv_dict[grp] = subj_means                              # for plotting
            else:  # avg_level == "all"
                group_mean = np.nanmean(grp_psd, axis=0)                  # ch × f
                err        = compute_err(grp_psd, err_method)
                indiv_dict[grp] = grp_psd                                 # epoch × ch × f

            mean_dict[grp] = group_mean
            err_dict[grp]  = err

        # cache results
        cache_key = (
            tuple(hue_cols),
            tuple(ch_names),
            method,
            avg_level,
            err_method,
            frozenset(kwargs.items()),
            None if bad_channels is None else frozenset((k, tuple(v)) for k, v in bad_channels.items()),
        )
        self.psd_results[cache_key] = dict(freq=freq, mean=mean_dict, err=err_dict)

        # plotting – one subplot per channel
        nrows, ncols = row_col_layout(n_channels)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
        axs = np.atleast_1d(axs).ravel()        # always 1-D iterator

        for ch_idx, (ax, ch_name) in enumerate(zip(axs, ch_names)):
            for clr, grp in zip(colours, unique_groups):
                label = (
                    ", ".join(f"{c}={v}" for c, v in zip(hue_cols, grp))
                    if isinstance(grp, tuple) else f"{hue_cols[0]}={grp}"
                )
                m = mean_dict[grp][ch_idx]
                ax.plot(freq, m, color=clr, linewidth=1.8, label=label)

                if err_dict[grp] is not None:
                    e = err_dict[grp][ch_idx]
                    ax.fill_between(freq, m-e, m+e, alpha=0.25, color=clr)

                # optional individual lines
                if plot_individual_points:
                    lines = indiv_dict[grp]                               # n × ch × f
                    for l in lines:
                        ax.plot(freq, l[ch_idx], color=clr, alpha=0.25, linewidth=0.8)

            ax.set(
                title=ch_name,
                xlabel="Frequency (Hz)",
                ylabel="PSD (mV/Hz)",
                xlim=(freq.min(), freq.max()),
            )
            ax.set_yscale("log")
            ax.label_outer()       # only keep outer tick-labels
            sns.despine(ax=ax)

        # single legend outside
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig, axs

    # Clustering methods
    def cluster_data(
        self,
        reducer=None,            # one of {None, 'umap', 'pca', 't-sne'}
        clusterer='kmeans',      # one of {'kmeans', 'hdbscan'}
        reducer_params=None,     # dict of parameters for the reducer
        clusterer_params=None,    # dict of parameters for the clusterer
        verbose=True
    ):
        """
            Clusters data based on an optional dimensionality reducer
            and a chosen clustering algorithm.

            TODO: Make it also a standalone method

            Parameters
            ----------
            data : pd.DataFrame or np.ndarray
                Already scaled feature data. Shape = [n_samples, n_features].
            reducer : str or None, optional
                Which dimensionality reducer to use
                ('umap', 'pca', 't-sne' or None).
                If None, no dimensionality reduction is applied.
            clusterer : str, optional
                Which clustering algorithm to use ('kmeans' or 'hdbscan').
            reducer_params : dict, optional
                Dictionary of hyperparameters for the reducer.
                Example for UMAP: {"n_neighbors": 15, "min_dist": 0.1,
                                    "n_components": 2, "random_state": 42}
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
            print("No reducer parameters provided. Using default values.")
        if clusterer_params is None:
            clusterer_params = {}
            print("No clusterer parameters provided. Using default values.")

        # 1. Dimensionality Reduction (if applied)
        if reducer == 'umap':
            self.reducer = umap.UMAP(**reducer_params)
            data_reduced = self.reducer.fit_transform(data)

        elif reducer == 'pca':
            self.reducer = PCA(**reducer_params)
            data_reduced = self.reducer.fit_transform(data)

        elif reducer == 't-sne':
            self.reducer = TSNE(**reducer_params)
            data_reduced = self.reducer.fit_transform(data)

        else:
            data_reduced = data
        self.dims = data_reduced.shape[1]
        self.dims_df = pd.DataFrame(
            data_reduced, columns=[f'dim{i+1}' for i in range(self.dims)]
            )

        # 2. Clustering
        if clusterer == 'kmeans':
            self.clusterer = KMeans(**clusterer_params)
            labels = self.clusterer.fit_predict(data_reduced)

        elif clusterer == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(**clusterer_params)
            labels = self.clusterer.fit_predict(data_reduced)

        else:
            raise ValueError("Unsupported clustering method."
                             "Choose from 'kmeans' or 'hdbscan'.")

        self.reducer_params = copy.deepcopy(reducer_params) if len(reducer_params) == 0 else copy.deepcopy(self.reducer.get_params())
        self.clusterer_params = copy.deepcopy(clusterer_params) if len(clusterer_params) == 0 else copy.deepcopy(self.clusterer.get_params())

        # Print results
        if verbose:
            print("Number of clusters:", len(np.unique(labels)))
            print(f"Percentage of clustered points: {np.sum(labels != -1) / len(labels) * 100}")

        self.labels = labels

    def clustering_grid_search(self, reducer, clusterer,
                               reducer_params, clusterer_params,
                               evaluation="metrics",
                               bayesian_metric="composite",
                               best_criterion="composite",
                               bayesian_iterations=25,
                               plot_clusters=False,
                               random_seed=42):
        """
        Perform a grid search over different clustering hyperparameters.

        Parameters
        ----------
        reducer : str or None
            The dimensionality reducer to use 
            (e.g. 'umap', 'pca', 't-sne' or None).
        clusterer : str
            The clustering algorithm to use ('kmeans' or 'hdbscan').
        reducer_params : dict
            Dictionary where each key is a reducer parameter name
            and its value is a list of options.
        clusterer_params : dict
            Dictionary where each key is a clusterer parameter name
            and its value is a list of options.
        evaluation : {"metrics", "bayesian"}, optional
            Which evaluation strategy to use.
            - "metrics": try every parameter combination and
                        evaluate with several clustering metrics.
            - "bayesian": use Bayesian optimization (numeric parameters only).
        bayesian_metric : str, optional
            For bayesian evaluation: which metric to optimize.
            Options include "silhouette","davies", "calinski",
            "n_clusters", "percent_clustered", or "composite".
            Default is "composite".
        best_criterion : str, optional
            Which metric to use for selecting the best parameter combination.
            Options: "silhouette", "davies", "calinski", "n_clusters",
            "percent_clustered", or "composite".
        bayesian_iterations : int, optional
            Number of Bayesian optimization iterations
            (only used if evaluation=="bayesian").

        Returns
        -------
        best_params : dict
            A dictionary with keys "reducer_params" and "clusterer_params"
            for the best-found combination.
        """
        import itertools
        from sklearn.metrics import silhouette_score, davies_bouldin_score, \
            calinski_harabasz_score

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # === Parameter Checks ===
        if self.feats is None:
            raise ValueError("No features found. Run get_features() first.")

        if not isinstance(reducer_params, dict):
            raise ValueError("reducer_params must be a dictionary.")
        if not isinstance(clusterer_params, dict):
            raise ValueError("clusterer_params must be a dictionary.")
        for key, val in reducer_params.items():
            if not isinstance(val, list) or len(val) == 0:
                raise ValueError(f"Reducer parameter '{key}'"
                                    "must be a non-empty list.")
        for key, val in clusterer_params.items():
            if not isinstance(val, list) or len(val) == 0:
                raise ValueError(f"Clusterer parameter '{key}'"
                                    "must be a non-empty list.")

        if evaluation not in ["metrics", "bayesian"]:
            raise ValueError("evaluation must be either"
                             "'metrics' or 'bayesian'.")
        if reducer is not None and reducer not in ["umap", "pca", "t-sne"]:
            raise ValueError("Reducer must be one of ['umap', 'pca', 't-sne']"
                             "or None.")
        if clusterer not in ["kmeans", "hdbscan"]:
            raise ValueError("Clusterer must be one of ['kmeans', 'hdbscan'].")

        allowed_bayesian_metrics = ["silhouette", "davies", "calinski",
                                    "n_clusters", "percent_clustered",
                                    "composite"]
        if bayesian_metric not in allowed_bayesian_metrics:
            raise ValueError("bayesian_metric must be one of"
                             f"{allowed_bayesian_metrics}.")

        allowed_metrics = ["silhouette", "davies", "calinski", "n_clusters",
                           "percent_clustered", "composite"]
        if best_criterion not in allowed_metrics:
            print(f"""Warning: best_criterion '{best_criterion}' not allowed.
                  Defaulting to 'composite'. Or select from:
                  {allowed_metrics}""")
            best_criterion = "composite"

        if evaluation == "bayesian":
            # Ensure all parameter values are numeric
            for param_dict in [reducer_params, clusterer_params]:
                for key, vals in param_dict.items():
                    for v in vals:
                        try:
                            float(v)
                        except Exception:
                            raise ValueError(
                                "Bayesian evaluation requires numeric values"
                                f"for parameter '{key}', got value {v}.")

        # === Grid Search Evaluation ===
        if evaluation == "metrics":
            results = []
            # Create all parameter combinations
            reducer_keys = list(reducer_params.keys())
            reducer_combos = list(itertools.product(*[reducer_params[k] for
                                                      k in reducer_keys]))
            clusterer_keys = list(clusterer_params.keys())
            clusterer_combos = list(itertools.product(*[clusterer_params[k] for
                                                        k in clusterer_keys]))

            # Initialize a plotting figure
            if plot_clusters:
                fig, ax = plt.subplots(len(reducer_combos),
                                       len(clusterer_combos),
                                       figsize=(5*len(clusterer_combos),
                                                3.5*len(reducer_combos)))

                # fix if axes has only 1 dimension
                if len(reducer_combos) == 1:
                    ax = ax[np.newaxis, :]
                if len(clusterer_combos) == 1:
                    ax = ax[:, np.newaxis]

            # Iterate over all combinations
            with tqdm(total=len(reducer_combos) * len(clusterer_combos),
                      desc="Grid Search") as pbar:
                for i, r_combo in enumerate(reducer_combos):
                    r_dict = dict(zip(reducer_keys, r_combo))
                    for j, c_combo in enumerate(clusterer_combos):
                        c_dict = dict(zip(clusterer_keys, c_combo))

                        temp_obj = copy.deepcopy(self)
                        try:
                            temp_obj.cluster_data(reducer=reducer,
                                                  clusterer=clusterer,
                                                  reducer_params={
                                                    **r_dict,
                                                    'random_state': random_seed},
                                                  clusterer_params=c_dict,
                                                  verbose=False)
                        except Exception as e:
                            print(f"Combination Reducer:{r_dict} &"
                                  f"Clusterer:{c_dict} failed: {e}")
                            continue

                        labels = temp_obj.labels
                        valid_clusters = [lab for lab in np.unique(labels) if
                                          lab != -1]
                        if len(valid_clusters) < 2:
                            sil = -1
                            davies = np.inf
                            calinski = -1
                        else:
                            data_used = temp_obj.dims_df.values if \
                                temp_obj.dims_df is not None else \
                                temp_obj.feats.values
                            sil = silhouette_score(data_used, labels)
                            davies = davies_bouldin_score(data_used, labels)
                            calinski = calinski_harabasz_score(data_used, labels)

                        n_clusters = len(valid_clusters)
                        percent_clustered = np.sum(
                            labels != -1) / len(labels) * 100

                        results.append({
                            **{f"reducer_{k}": v for k, v in r_dict.items()},
                            **{f"clusterer_{k}": v for k, v in c_dict.items()},
                            "param_combo": f"Reducer:{r_dict}, Clusterer:{c_dict}",
                            "silhouette": sil,
                            "davies": davies,
                            "calinski": calinski,
                            "n_clusters": n_clusters,
                            "percent_clustered": percent_clustered
                        })

                        if plot_clusters:
                            temp_obj.plot_umap(n_components=2,
                                               ax=ax[i, j],
                                               reducer_params={
                                                   **r_dict,
                                                   "random_state": random_seed},
                                               palette="Set2")

                            reducer_text = "\n".join([f"{k}: {v}" for
                                                      k, v in r_dict.items()])
                            clusterer_text = "\n".join([f"{k}: {v}" for
                                                        k, v in c_dict.items()])

                            textbox = f"""Reducer:\n{reducer_text}\n
                                        Clusterer:\n{clusterer_text}"""

                            ax[i, j].text(1.05, 0.025, textbox,
                                          transform=ax[i, j].transAxes,
                                          fontsize=8,
                                          verticalalignment='bottom',
                                          horizontalalignment='left',
                                          bbox=dict(facecolor='white', alpha=0.7)
                                          )
                        pbar.update(1)

            results_df = pd.DataFrame(results)
            if results_df.empty:
                print("No valid clustering combinations found.")
                return None, None

            # Normalize metrics for composite score (for metrics where higher is better)
            for metric in ["silhouette", "calinski", 
                           "n_clusters", "percent_clustered"]:
                mn = results_df[metric].min()
                mx = results_df[metric].max()
                if mx - mn != 0:
                    results_df[f"norm_{metric}"] = (
                        results_df[metric] - mn) / (mx - mn
                                                    )
                else:
                    results_df[f"norm_{metric}"] = 0
            # For Davies–Bouldin, lower is better so invert the normalization
            mn = results_df["davies"].min()
            mx = results_df["davies"].max()
            if mx - mn != 0:
                results_df["norm_davies"] = (
                    mx - results_df["davies"]) / (mx - mn)
            else:
                results_df["norm_davies"] = 0

            # Composite score (equal weighting)
            results_df["composite"] = (
                results_df["norm_silhouette"] +
                results_df["norm_davies"] +
                results_df["norm_calinski"] +
                results_df["norm_n_clusters"] +
                results_df["norm_percent_clustered"]
            ) / 5

            best_idx = results_df[best_criterion].idxmax()
            best_params = {
                "reducer_params": {k.replace("reducer_", ""): results_df.loc[best_idx, k]
                                     for k in results_df.columns if k.startswith("reducer_")},
                "clusterer_params": {k.replace("clusterer_", ""): results_df.loc[best_idx, k]
                                     for k in results_df.columns if k.startswith("clusterer_")}
            }

            if plot_clusters:
                plt.tight_layout()

        # === Bayesian Optimization Evaluation ===
        elif evaluation == "bayesian":
            from bayes_opt import BayesianOptimization

            # Build numeric search space from union of reducer and clusterer params
            search_space = {}
            all_params = {**reducer_params, **clusterer_params}
            for key, vals in all_params.items():
                numeric_vals = [float(v) for v in vals]
                search_space[key] = (min(numeric_vals), max(numeric_vals))

            def objective(**params):
                # Map continuous parameters to the closest option
                r_dict = {}
                c_dict = {}
                for key, val in params.items():
                    if key in reducer_params:
                        r_dict[key] = min(reducer_params[key],
                                          key=lambda x: abs(float(x) - val))
                    elif key in clusterer_params:
                        c_dict[key] = min(clusterer_params[key],
                                          key=lambda x: abs(float(x) - val))
                temp_obj = copy.deepcopy(self)
                try:
                    temp_obj.cluster_data(reducer=reducer,
                                          clusterer=clusterer,
                                          reducer_params=r_dict,
                                          clusterer_params=c_dict,
                                          verbose=False)
                except Exception:
                    return -1e6  # heavy penalty

                labels = temp_obj.labels
                valid_clusters = [lab for lab in np.unique(labels) if lab != -1]
                if len(valid_clusters) < 2:
                    sil = -1
                    davies = np.inf
                    calinski = -1
                else:
                    data_used = temp_obj.dims_df.values if \
                        temp_obj.dims_df is not None else \
                        temp_obj.feats.values
                    sil = silhouette_score(data_used, labels)
                    davies = davies_bouldin_score(data_used, labels)
                    calinski = calinski_harabasz_score(data_used, labels)

                n_clusters = len(valid_clusters)
                percent_clustered = np.sum(labels != -1) / len(labels) * 100

                if bayesian_metric == "silhouette":
                    return sil
                elif bayesian_metric == "davies":
                    return -davies
                elif bayesian_metric == "calinski":
                    return calinski
                elif bayesian_metric == "n_clusters":
                    return n_clusters
                elif bayesian_metric == "percent_clustered":
                    return percent_clustered
                else:  # composite
                    composite = sil + (1.0 / (davies + 1e-6))
                    + (calinski / 1000.0)
                    + (percent_clustered / 100) 
                    + n_clusters
                    return composite

            optimizer = BayesianOptimization(
                f=objective,
                pbounds=search_space,
                verbose=2,
                random_state=42
            )
            optimizer.maximize(init_points=5, n_iter=bayesian_iterations)

            best_params_cont = optimizer.max["params"]
            best_reducer_params = {}
            best_clusterer_params = {}
            for key, val in best_params_cont.items():
                if key in reducer_params:
                    best_reducer_params[key] = min(reducer_params[key],
                                                   key=lambda x: abs(float(x) - val))
                elif key in clusterer_params:
                    best_clusterer_params[key] = min(clusterer_params[key],
                                                     key=lambda x: abs(float(x) - val))
            best_params = {"reducer_params": best_reducer_params,
                           "clusterer_params": best_clusterer_params}

            # Build a simple results dataframe from the Bayesian iterations
            bayes_results = []
            for res in optimizer.res:
                params_used = res["params"]
                r_dict = {}
                c_dict = {}
                for key, val in params_used.items():
                    if key in reducer_params:
                        r_dict[key] = min(reducer_params[key],
                                          key=lambda x: abs(float(x)-val))
                    elif key in clusterer_params:
                        c_dict[key] = min(clusterer_params[key],
                                          key=lambda x: abs(float(x)-val))
                bayes_results.append({
                    "param_combo": f"Reducer: {r_dict}, Clusterer: {c_dict}",
                    bayesian_metric: res["target"]
                })
            results_df = pd.DataFrame(bayes_results)

        # Save the results dataframe in self for later plotting
        self.grid_search_results_df = results_df.copy()
        return best_params, results_df

    def plot_grid_search_results(self, metric="composite", ascending=True):
        """
        Plot the grid search results stored in self.grid_search_results_df.

        Parameters
        ----------
        metric : str or list, optional
            The metric name(s) to plot. Allowed metrics include:
            "silhouette", "davies", "calinski", "n_clusters",
            "percent_clustered", "composite".
            If a list is provided, a separate plot is generated for each metric.
        sort_order : str, optional
            Sort order for the barplot. Can be "ascending" or "descending".

        Returns
        -------
        None
        """
        if not hasattr(self, "grid_search_results_df"):
            raise ValueError("No grid search results found."
                             "Run clustering_grid_search() first.")

        df = self.grid_search_results_df.copy()
        allowed_metrics = ["silhouette", "davies", "calinski",
                           "n_clusters", "percent_clustered", "composite"]

        if isinstance(metric, str):
            if metric not in allowed_metrics:
                raise ValueError(f"Metric '{metric}' is not among allowed"
                                 f"metrics: {allowed_metrics}.")
            sorted_df = df.sort_values(by=metric, ascending=ascending)
            plt.figure(figsize=(10, max(6, len(sorted_df) * 0.3)))
            plt.barh(sorted_df["param_combo"], sorted_df[metric])
            plt.xlabel(metric)
            plt.title(f"Grid Search Results - {metric}")
            plt.tight_layout()
            plt.show()
        elif isinstance(metric, list):
            for met in metric:
                if met not in allowed_metrics:
                    raise ValueError(f"Metric '{met}' is not among allowed"
                                     f"metrics: {allowed_metrics}.")
            n = len(metric)
            fig, axs = plt.subplots(n, 1, figsize=(10, 4 * n))
            if n == 1:
                axs = [axs]
            for ax, met in zip(axs, metric):
                sorted_df = df.sort_values(by=met, ascending=ascending)
                ax.barh(sorted_df["param_combo"], sorted_df[met])
                ax.set_xlabel(met)
                ax.set_title(f"Grid Search Results - {met}")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Parameter 'metric' must be a string or"
                             "a list of strings.")

    def plot_dim_reduction(self, dims=(1, 2), ax=None, plot3d=False,
                           plot_outliers=True, plot_labels=True,
                           palette='tab10', edgecolor='black',
                           s=10, alpha=0.5, figsize3d=(10, 10),
                           display=False, **kwargs):
        """
            Plot the dimensionaliry reduction results from self.dims_df
        """
        if self.dims_df is None:
            raise ValueError("No dimensionality reduction results found."
                             "Run cluster_data() first.")
        
        plot_df = self.dims_df.copy()
        if self.labels is not None:
            plot_df['label'] = self.labels
        
        if not plot_outliers:
            plot_df = plot_df[plot_df['label'] != -1]

        if not all(f'dim{dim}' in self.dims_df.columns for dim in dims):
            raise ValueError(f"One or more specified dimensions {dims} do not"
                             "exist in the dimensionality reduction results."
                             f"Current availabel dimensions {self.dims_df.columns}.")

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
            if display:
                plt.show()

    # Feature vizualization methods

    def plot_simple_pca(self, n_components=2, x='PC1', y='PC2', plot_outliers=True, title='PCA',
                        ax=None, figisize=(4,4), palette='tab10', edgecolor='black',
                        display=False, **kwargs):
                            
        """
            Perform and visualize PCA.

            Parameters:
            -----------
            n_components : int, optional (default=2)
                Number of principal components to compute.
            x : str, optional (default='PC1')
                The principal component to plot on the x-axis.
            y : str, optional (default='PC2')
                The principal component to plot on the y-axis.
            title : str, optional (default='PCA')
                Title of the plot.
            ax : matplotlib.axes.Axes, optional (default=None)
                The axes on which to plot. If None, a new figure and axes will be created.
            figisize : tuple, optional (default=(4,4))
                Size of the figure.
            palette : str or sequence, optional (default='tab10')
                Colors to use for the different levels of the hue variable.
            edgecolor : str, optional (default='black')
                Color of the edge of the points.
            **kwargs : additional keyword arguments
                Additional arguments to pass to the PCA constructor.
            Returns:
            --------
            None
                This function does not return anything. It displays a PCA plot.
            Notes:
            ------
            This function performs Principal Component Analysis (PCA) on the features of the object and 
            visualizes the first two principal components in a scatter plot. If labels are provided, 
            they are used to color the points in the plot.

        """

        pca = PCA(n_components=n_components, **kwargs)
        pca.fit(self.feats)
        X_pca = pca.transform(self.feats)

        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['label'] = self.labels
        hue = 'label' if self.labels is not None else None

        if not plot_outliers and self.labels is not None:
            pca_df = pca_df[pca_df['label'] != -1]

        # Plot the PCA results
        if ax is None:
            fig, ax = plt.subplots(figsize=figisize)

        sns.scatterplot(data=pca_df, x=x, y=y , hue=hue, 
                        palette=palette, ax=ax, legend=True, edgecolor = edgecolor)
        ax.set_title(title)

        if self.labels is not None:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Label')
        if display:
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
                sns.scatterplot(data=plot_df, x='t-SNE 1', y='t-SNE 2', hue='label', ax=ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Label')
            else:
                sns.scatterplot(data=plot_df, x='t-SNE 1', y='t-SNE 2', ax=ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            return ax  


    def plot_umap(self, n_components, reducer_params={}, ax=None, plot3d=False, plot_outliers=True,
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
        if 'n_components' not in reducer_params:
            reducer_params['n_components'] = n_components
        reducer = umap.UMAP(**reducer_params)

        embedding = reducer.fit_transform(self.feats)

        plot_df = pd.DataFrame(embedding, columns=[f'UMAP {i+1}' for i in range(embedding.shape[1])])

        if self.labels is not None:
            plot_df['label'] = self.labels
        
        if not plot_outliers:
            plot_df = plot_df[plot_df['label'] != -1]

        if plot3d and reducer_params['n_components'] >= 3:
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
                sns.scatterplot(data=plot_df, x='UMAP 1', y='UMAP 2', hue='label', ax = ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, legend=True, **kwargs)
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Label')
            else:
                sns.scatterplot(data=plot_df, x='UMAP 1', y='UMAP 2', ax=ax, palette=palette, edgecolor=edgecolor, s=s, alpha=alpha, **kwargs)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')


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

    def clone(self):
        """
            Create a deep copy of the Epochs object.

            Returns:
            --------
            Epochs
                A deep copy of the current Epochs object.
        """
        return copy.deepcopy(self)

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

    @classmethod
    def concatenate(cls, epochs_list, *, reset_index=True):
        """
        Concatenate several *et.Epochs* objects (wrapper instances) into
        **one** new *et.Epochs* object.

        Parameters
        ----------
        epochs_list : list[Epochs]
            The objects to concatenate - order is preserved.
        reset_index : bool, default ``True``
            If *True* every metadata DataFrame is ``reset_index(drop=True)``
            **before** concatenation.

        Returns
        -------
        Epochs
            A new instanced with concatenated ``mne.Epochs`` and
            merged metadata.  ``non_feature_cols`` is the **union** of the
            individual objects.  All other scratch attributes start empty.
        """
        if not epochs_list:
            raise ValueError("epochs_list is empty.")
        if not all(isinstance(e, cls) for e in epochs_list):
            raise TypeError("All elements must be instances of Epochs.")

        meta_frames = []
        mne_epochs   = []
        all_feats = pd.DataFrame()
        union_non_feat = set()
        for ep in epochs_list:
            # keep track of which animal/condition the rows came from
            meta = ep.metadata.copy()
            if reset_index:
                meta = meta.reset_index(drop=True)
            meta_frames.append(meta)
            mne_epochs.append(ep.epochs)
            union_non_feat.update(ep.non_feature_cols)
            all_feats = pd.concat([all_feats, ep.feats], ignore_index=True)

        # concatenate mne.Epochs objects
        concat_mne   = mne.concatenate_epochs(mne_epochs)
        concat_meta  = pd.concat(meta_frames, ignore_index=True)
        concat_mne.metadata = concat_meta

        new_obj = cls(
            concat_mne,
            non_feature_cols=list(union_non_feat),
            animal_id=None,
            condition=None,
        )
        new_obj.feats = all_feats.reset_index(drop=True)
        return new_obj


