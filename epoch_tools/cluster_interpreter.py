import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages

import shap  # pip install shap
import os

from .utils import row_col_layout

class ClusterInterpreter:
    """
    A helper class to interpret clusters from an epoch_tools.Epochs object. 
    Trains a supervised model on the cluster labels (as pseudo-labels)
    and provides multiple visualization methods:
      - Confusion matrix and accuracy
      - Global feature importances (model-wide)
      - SHAP-based per-cluster interpretability
    """

    def __init__(self,
                 epochs,
                 model_type='random_forest', 
                 model_params=None,
                 test_size=0.2,
                 random_state=42):
        """
        Initialize a ClusterInterpreter object.

        Parameters:
        -----------
        epochs : epoch_tools.Epochs
            The Epochs object containing features and cluster labels.
        model_type : str, optional
            Type of classifier to use ('random_forest', 'xgboost', 'catboost').
        model_params : dict, optional
            Hyperparameters for the chosen model.
        test_size : float, optional
            Fraction of data to use for testing.
        random_state : int, optional
            Random seed for reproducibility.
        """
        if model_params is None:
            model_params = {} #TODO: Default hyperparameters for each model type

        self.epochs = epochs
        self.model_type = model_type
        self.model_params = model_params
        self.test_size = test_size
        self.random_state = random_state
        
        # Will be set once fit() is called
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def fit(self):
        """
        Fit the chosen model on epoch_tools.Epochs' features (epochs.feats) to predict cluster labels (epochs.labels).

        Parameters
        ----------
        epochs : epoch_tools.Epochs
            The epoch_tools.Epochs object that has:
               - epochs.feats : DataFrame of features 
               - epochs.labels : array-like cluster assignments
        """
        if self.epochs.feats is None:
            raise ValueError("No features found. Make sure you call TainiEpochs.get_features() first.")
        if not hasattr(self.epochs, 'labels') or self.epochs.labels is None:
            raise ValueError("No cluster labels found. Make sure you've assigned `epochs.labels` from clustering.")

        self.X = self.epochs.feats
        self.y = self.epochs.labels

        # Split into train/test for validation (even though they're pseudo-labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y  # keeps cluster distribution consistent
        )

        # Create the model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=self.random_state, 
                **self.model_params
            )
        elif self.model_type == 'gbm':
            self.model = GradientBoostingClassifier(
                random_state=self.random_state, 
                **self.model_params
            )
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=self.random_state, **self.model_params)
        elif self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(random_state=self.random_state, **self.model_params)
        elif self.model_type == 'catboost':
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(random_state=self.random_state, verbose=0, **self.model_params)
        else:
            raise ValueError("Invalid model_type. Choose 'random_forest', 'xgboost', 'lightgbm', or 'catboost'.")
        
        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        self.y_pred = self.model.predict(self.X_test)

        print(f"Model trained on {len(self.X_train)} samples. Held out {len(self.X_test)} for testing.")

    def plot_confusion_matrix(self, normalize=False, annot=True, cmap="Blues", ax=None, figsize=(5, 5), display=True):
        """
        Plot the confusion matrix of predicted labels on the test set.

        Parameters
        ----------
        normalize : bool
            Whether to normalize counts to [0,1].
        """
        if self.model is None or self.y_pred is None:
            raise ValueError("Model not trained or y_pred not available. Call `fit(...)` first.")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
    
        sns.heatmap(cm, annot=annot, cmap=cmap,
                    xticklabels=self.model.classes_, yticklabels=self.model.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        
        if display:
            plt.tight_layout()
            plt.show()
        return ax

    def get_accuracy(self):
        """
        Compute the accuracy of the model on the test set.
        """
        if self.model is None or self.y_pred is None:
            raise ValueError("Model not trained or y_pred not available. Call `fit(...)` first.")

        return accuracy_score(self.y_test, self.y_pred)

    def plot_feature_importances_global(self, top_n=20, ax=None, figsize=(8, 5), display=True):
        """
        Plot the global (model-wide) feature importances from the trained model (if available).
        This does NOT distinguish among clusters; it's the overall importance in the multi-class setting.

        Parameters
        ----------
        top_n : int
            Number of top features to display in the plot.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call `fit(...)` first.")

        # Many sklearn ensemble models have `feature_importances_`
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            raise ValueError(f"Feature importances are not available for model '{self.model_type}'.")

        feature_names = self.X_train.columns
        idxs = np.argsort(importances)[::-1]
        top_features = idxs[:top_n]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x=importances[top_features],
            y=[feature_names[i] for i in top_features],
            orient="h",
            ax=ax
        )
        ax.set_title(f"Top {top_n} Global Feature Importances ({self.model_type})")
        ax.set_xlabel("Importance")
        if display:
            plt.tight_layout()
            plt.show()
    

    def plot_feature_values(self, exclude_features = [], plot_outliers=True, plot_type='violin', 
                            palette='tab10', ax=None, figsize=None, display=True, **kwargs):
        """
        Visualize feature distributions for different clusters.

        Parameters:
        -----------
        plot_type : str, optional
            Type of plot ('violin' or 'box').
        display : bool, optional
            Whether to display the plot immediately.
        palette : str, optional
            Color palette for clusters.
        """
        
        data = self.epochs.feats.copy(deep=True)
        data['labels'] = self.epochs.labels
        # Exclude features if specified
        if exclude_features:
            data = data.drop(columns=exclude_features)
        
        if not plot_outliers:
            data = data[data['labels'] != -1]

        melt_df = data.melt(id_vars='labels', value_vars=data.columns, 
                            var_name='Feature', value_name='Value')
        
        n_features = len(data.columns) - 1
        # figsize width scales with number of features
        if ax is None:
            figsize = (n_features, 6) if figsize is None else figsize
            fig, ax = plt.subplots(figsize=figsize)

        if plot_type == 'violin':
            sns.violinplot(x='Feature', y='Value', hue='labels', data=melt_df, ax=ax, palette=palette, **kwargs)
        elif plot_type == 'box' or plot_type == 'boxplot':
            sns.boxplot(x='Feature', y='Value', hue='labels', data=melt_df, ax=ax, palette=palette, **kwargs)
        else:
            raise ValueError("Invalid plot_type. Choose 'violin' or 'box'.")
        
        ax.set_ylabel('Scaled Feature Value')
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(f'Feature Values per Cluster')
        ax.legend(title="Cluster", loc='center left', bbox_to_anchor=(1, 0.5))

        if display:
            plt.tight_layout()
            plt.show()
        return ax


    def plot_feature_density(self, common_norm=False, exclude_features = [], plot_outliers=True,
                             palette='tab10', figsize=None, display=True, **kwargs):
        """
        Plot kernel density estimations for feature distributions across clusters.

        Parameters:
        -----------
        display : bool, optional
            Whether to display the plot immediately.
        palette : str, optional
            Color palette for clusters.
        common_norm : bool, optional
            Whether to normalize density estimates across clusters.
        """    
        
        data = self.epochs.feats.copy(deep=True)
        # Exclude features if specified
        if exclude_features:
            data = data.drop(columns=exclude_features)
        features = data.columns
        n_features = len(features)
        data['labels'] = self.epochs.labels

        if not plot_outliers:
            data = data[data['labels'] != -1]
    
        nrows, ncols = row_col_layout(n_features)
        figsize = (ncols*3.5, nrows*3.5) if figsize is None else figsize
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        ax = ax.ravel()

        for i, feature in enumerate(features):
            sns.kdeplot(data=data, x=feature, hue='labels', ax=ax[i], fill=True,
                        legend=True, hue_order=np.sort(data['labels'].unique()),
                         palette=palette, common_norm=common_norm, **kwargs)
            ax[i].set_title(feature)
            ax[i].set_xlabel('Feature Value')
            # Determine row and column of the current subplot
            row, col = divmod(i, ncols)
            if col == ncols - 1:
                handles, labels = ax[i].get_legend_handles_labels()
                if handles:
                    ax[i].legend(handles, labels, loc='upper left', bbox_to_anchor=(1.1, 1), title='Label')
            else:
                # Remove the legend for subplots not in the last column.
                leg = ax[i].get_legend()
                if leg is not None:
                    leg.remove()

        # Remove unused axes
        for j in range(i+1, len(ax)):
            fig.delaxes(ax[j])
            
        plt.tight_layout()
        if display:
            plt.show()


    def plot_epoch_signal(self, n_epochs=5, plot_outliers=True, channels_to_plot='all', cmap='hls', nstd='auto', display=True):
        """
        Plot raw EEG signals for epochs from different clusters.

        Parameters:
        -----------
        n_epochs : int, optional
            Number of epochs to visualize per cluster.
        channels_to_plot : list or 'all', optional
            Channels to include in the plot.
        cmap : str or list, optional
            Color map for channels.
        nstd : int, optional (default='auto')
            Number of standard deviations to plot around the mean for y-axis limits
        display : bool, optional
            Whether to display the plot immediately.
        """
        
        import matplotlib.gridspec as gridspec

        if channels_to_plot == 'all':
            channels_to_plot = self.epochs.ch_names
        labels = self.epochs.labels
        if not plot_outliers:
            labels = labels[labels != -1]
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        if isinstance(cmap, str):
            colors = sns.color_palette(cmap, len(channels_to_plot))
        elif isinstance(cmap, list):
            colors = cmap

        # Create a figure without pre-defined axes
        fig = plt.figure(figsize=(6 * n_labels, 4 * n_epochs))

        # Loop through the unique labels and plot for each
        for i, label in enumerate(unique_labels):
            # Get the indices of epochs corresponding to the current label
            label_ids = np.array(self.epochs.feats[self.epochs.labels == label].index)

            random_ids = np.random.choice(label_ids, n_epochs, replace=False)

            for j, epoch_id in enumerate(random_ids):
                # Define GridSpec for the current (j, i) cell
                gs = gridspec.GridSpecFromSubplotSpec(
                    len(channels_to_plot), 1,  # Split into len(channels_to_plot) rows
                    subplot_spec=gridspec.GridSpec(n_epochs, n_labels)[j, i],  # GridSpec for this subplot
                    hspace=0.1 #just spacing between rows
                )

                for k, channel in enumerate(channels_to_plot):
                    # Create a subplot for the specific channel
                    ax_channel = fig.add_subplot(gs[k, 0])
                    ax_channel.plot(self.epochs.get_data(picks=channel)[epoch_id][0], color=colors[k], linewidth=0.9)
                    ax_channel.set_title(f"Epoch {epoch_id} - Cluster {label}")
                    ax_channel.set_ylabel(channel,rotation=0, ha='right')
                    ax_channel.set_xlabel('Time (samples)')
                    ax_channel.set_yticklabels([])

                    # Channel y-axis limits as nstd from mean
                    if nstd != 'auto':
                        _d = self.epochs.get_data(picks=channel)[:, 0, :]
                        _mean = np.mean(_d)
                        _std = np.std(_d)
                        ax_channel.set_ylim(_mean - nstd*_std, _mean + nstd*_std)

                    if k == 0:
                        if len(channels_to_plot) > 1:
                            ax_channel.spines['bottom'].set_visible(False)
                        ax_channel.set_xticks([])
                        ax_channel.set_xlabel("")
                    elif k == len(channels_to_plot) - 1:
                        ax_channel.spines['top'].set_visible(False)
                        ax_channel.set_title("")
                    else:
                        ax_channel.set_title("")
                        ax_channel.set_xlabel("")
                        ax_channel.set_xticks([])
                        ax_channel.spines['top'].set_visible(False)
                        ax_channel.spines['bottom'].set_visible(False)    
        plt.tight_layout()
        if display:
            plt.show()


    def plot_feature_hierarchy(self, plot_outliers=True, method='ward', cmap='coolwarm', 
                            row_cluster=False, col_cluster=True,
                            figsize='auto', vmin=-3, vmax=3,
                            col_colors='labels', display=True):
        """
        Generate a hierarchical clustering heatmap for features.

        Parameters:
        -----------
        method : str, optional
            Linkage method for hierarchical clustering.
        cmap : str, optional
            Color map for heatmap.
        row_cluster : bool, optional
            Whether to cluster rows.
        col_cluster : bool, optional
            Whether to cluster columns.
        figsize : tuple or 'auto', optional
            Figure size.
        vmin, vmax : float, optional
            Color scale limits.
        col_colors : str or pd.Series, optional
            Cluster label colors.
        display : bool, optional
            Whether to display the plot immediately.
        """
        data = self.epochs.feats.copy(deep=True)
        n_features = len(data.columns)
        data['Cluster'] = self.epochs.labels

        if not plot_outliers:
            data = data[data['Cluster'] != -1]

        # Color labels
        if col_colors == 'labels':
            unique_labels = np.unique(data['Cluster'].unique())
            colors = sns.color_palette("tab10", n_colors=len(unique_labels))
            col_colors = pd.Series(
                data['Cluster'].map(dict(zip(unique_labels, colors)))
            )
            # Create a legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=colors[i], markersize=10, 
                            label=f'Label {label}')
                    for i, label in enumerate(unique_labels)]
            legend_title = "Cluster"
            
        elif isinstance(col_colors, pd.Series):
            # If col_colors is a Series, assume it's already mapped to colors
            unique_labels = col_colors.unique()
            colors = col_colors.drop_duplicates().values
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color, markersize=10, 
                            label=f'Label {label}')
                    for label, color in zip(unique_labels, colors)]
            legend_title = "Cluster"
        else:
            raise ValueError("Invalid col_colors. Must be 'labels' or a pd.Series.")

        sns.clustermap(data.drop(columns='Cluster').T,
                       method=method,
                       cmap=cmap,
                       row_cluster=row_cluster,
                       col_cluster=col_cluster,
                       figsize=(8, n_features*0.5) if figsize == 'auto' else figsize,
                       vmin=vmin, vmax=vmax,
                       col_colors=col_colors)

        # Add legend
        plt.legend(handles=legend_elements, title=legend_title, 
                   loc='upper left', bbox_to_anchor=(0, 0))

        if display:
            plt.show()


    # SHAP
    def get_shap_values(self, n_samples=2000):
        """
        Compute SHAP values for the trained model on the test set.

        Parameters
        ----------
        n_samples : int
            Number of samples to use for SHAP computation.
        """
        if self.model is None or self.y_pred is None:
            raise ValueError("Model not trained or y_pred not available. Call `fit(...)` first.")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X[:n_samples])

        return explainer, shap_values

    def plot_shap_summary(self, 
                          cluster=None,
                          n_samples=2000, 
                          max_display=10, 
                          plot_type="dot", 
                          figsize=None, 
                          display=True,
                          **kwargs):
        """
        Generate a SHAP summary plot for feature importance.

        Parameters:
        -----------
        cluster : int or None, optional
            If specified, show SHAP values for a single cluster.
            Otherwise, show all clusters.
        n_samples : int, optional
            Number of samples to compute SHAP values.
        max_display : int, optional
            Maximum number of features to display.
        plot_type : str, optional
            Type of SHAP plot ('dot', 'bar', etc.).
        figsize : tuple, optional
            Figure size.
        display : bool, optional
            Whether to display the plot immediately.
        """


        explainer, shap_values = self.get_shap_values(n_samples=n_samples)
        n_clusters = shap_values.shape[2]

        if cluster is not None:

            # Adjust cluster index if -1 exist in the labels
            cluster_i = cluster+1 if -1 in self.y else cluster
            cluster_shap_vals = shap_values[:, :, cluster_i]

            if figsize is None:
                figsize = (10, 4)
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            
            plt.sca(axes)
            shap.summary_plot(cluster_shap_vals, self.X[:n_samples], plot_type=plot_type, 
                    max_display=max_display, show=False, **kwargs)
            axes.set_title(f"SHAP Summary - Cluster {cluster}", fontsize=14)
            if display:
                plt.show()
        else:
            if figsize is None:
                figsize = (10, 3 * n_clusters)
            fig, axes = plt.subplots(n_clusters,1, figsize=figsize)
            axes = axes.flatten()


            for cluster_idx in range(n_clusters):
                cluster_shap_vals = shap_values[:, :, cluster_idx]
                cluster_label = cluster_idx-1 if -1 in self.y else cluster_idx

                # Set current axis so shap.summary_plot() draws there
                plt.sca(axes[cluster_idx])

                shap.summary_plot(
                    cluster_shap_vals,
                    self.X[:n_samples], 
                    plot_type=plot_type,
                    plot_size=None,
                    max_display=max_display,
                    show=False,
                    **kwargs
                )
                
                # You can set a custom subplot title here
                axes[cluster_idx].set_title(f"Cluster {cluster_label}", fontsize=12)

            plt.tight_layout()
            if display:
                plt.show()

    def plot_shap_decision_plot(
        self,
        cluster=None,
        n_samples=1000,
        n_shap_samples=2000,
        feature_names=None,
        link='identity',
        figsize=None,
        display=True,
        **kwargs
    ):
        """
        Generate a SHAP decision plot for model predictions.

        Parameters:
        -----------
        cluster : int or None, optional
            If specified, show SHAP decision plot for a single cluster.
        n_samples : int, optional
            Number of samples to display.
        n_shap_samples : int, optional
            Number of samples used to compute SHAP values.
        feature_names : list or None, optional
            Feature names for the plot.
        link : str, optional
            Link function for SHAP values.
        figsize : tuple, optional
            Figure size.
        display : bool, optional
            Whether to display the plot immediately.
        """
        explainer, shap_values = self.get_shap_values(n_samples=n_shap_samples)
        n_clusters = shap_values.shape[2]
        n_features = shap_values.shape[1]
        
        if cluster is not None:
            if cluster >= n_clusters:
                raise ValueError(f"Requested cluster={cluster} exceeds the number of clusters={n_clusters}")
            
            cluster_i = cluster+1 if -1 in self.y else cluster
            shap_values = shap_values[:, :, cluster_i]  # shape (n_shap_samples, n_features)

            if figsize is None:
                figsize = (10, 5)
            fig, axes = plt.subplots(1, 1, figsize=figsize)

            plt.sca(axes)
            shap.decision_plot(
                explainer.expected_value[cluster],
                shap_values[:n_samples],
                self.X[:n_samples],
                feature_names=feature_names,
                link=link,
                show=False,  # to allow capturing or customizing the plot
                **kwargs
            )
            axes.set_title(f"SHAP Decision Plot (Cluster {cluster})", fontsize=14)

            plt.tight_layout()
            if display:
                plt.show()

        else:
            if figsize is None:
                figsize = (10, 4 * n_clusters)
            fig, axes = plt.subplots(n_clusters, 1, figsize=figsize)
            axes = axes.flatten()

            for cluster_idx in range(n_clusters):
                cluster_shap_vals = shap_values[:, :, cluster_idx]

                cluster_label = cluster-1 if -1 in self.y else cluster

                # Set current axis so shap.summary_plot() draws there
                plt.sca(axes[cluster_idx])

                shap.decision_plot(
                    explainer.expected_value[cluster_idx],
                    cluster_shap_vals,
                    self.X[:n_samples],
                    feature_names=feature_names,
                    link=link,
                    auto_size_plot=None,
                    show=False,  # to allow capturing or customizing the plot
                    **kwargs
                )
                # You can set a custom subplot title here
                axes[cluster_idx].set_title(f"Cluster {cluster_label}", fontsize=12)

            plt.tight_layout()
            if display:
                plt.show()


    def plot_shap_dependence_plot(self,
                                feature,
                                interaction_feature=None,
                                cluster=None,
                                n_shap_samples=2000,
                                figsize=(6, 5),
                                display=True):
        """
        Plots a SHAP dependence plot for a given feature (and optional interaction feature).

        Parameters
        ----------
        feature : str
            The main feature for the x-axis.
        interaction_feature : str, optional
            Feature to color the points by. If None, SHAP will pick the strongest interaction.
        cluster : int, optional
            If specified, only plot SHAP values (and data) for the given cluster label.
        n_shap_samples : int
            Number of samples used to compute shap_values in the first place.
        display : bool
            If True, display the plot immediately (plt.show()).
            Otherwise, return the figure object for further manipulation.
        """

        explainer, shap_values_3d = self.get_shap_values(n_samples=n_shap_samples)
        n_classes = shap_values_3d.shape[2]


        if cluster is not None:
            if cluster >= n_classes:
                raise ValueError(f"Requested cluster={cluster} exceeds the number of classes={n_classes}")
            cluster_i = cluster+1 if -1 in self.y else cluster
            shap_values = shap_values_3d[:, :, cluster_i]  # shape (n_shap_samples, n_features)
        else:
            # For multi-class, you can either pick a single cluster or average across classes
            # or pass them all. shap.dependence_plot typically needs a 2D array, so let's
            # average across classes for a global perspective:
            shap_values = shap_values_3d.mean(axis=2)  # shape (n_shap_samples, n_features)

        # Subset the data if user wants a specific cluster
        X_sample = self.X[:n_shap_samples]
        if cluster is not None:
            cluster_mask = (self.y[:n_shap_samples] == cluster)
            X_sample = X_sample[cluster_mask]
            shap_values = shap_values[cluster_mask]
            if X_sample.empty:
                raise ValueError(f"No samples found for cluster={cluster} in the first {n_shap_samples} rows.")

        
        fig = plt.figure(figsize=figsize)
        shap.dependence_plot(
            ind=feature,
            shap_values=shap_values,
            features=X_sample,
            interaction_index=interaction_feature,
            show=False  # to allow capturing or customizing the plot
        )
        plt.set_title(f"SHAP Dependence Plot on '{feature}'" +
                (f" (Cluster {cluster})" if cluster is not None else ""), fontsize=14)

        if display:
            plt.show()

        
    def generate_report(self, methods, filename="report.pdf", method_configs=None, overwrite=False):
        """
        Runs each specified method in sequence, capturing each figure in a PDF.

        Parameters
        ----------
        methods : list of str
            Names of the methods in this class to call for the report.
        filename : str
            Where to save the resulting PDF.
        method_configs : dict or None
            A dictionary mapping method names to a dict of keyword arguments.
            Example:
                {
                    "plot_confusion_matrix": {"normalize": True},
                    "plot_feature_importances_global": {"top_n": 10}
                }
        """
        if methods == 'all':
            methods = [
                'plot_confusion_matrix',
                'plot_feature_importances_global',
                'plot_feature_values',
                'plot_feature_density',
                'plot_epoch_signal',
                'plot_feature_hierarchy',
                'plot_shap_summary',
                'plot_shap_decision_plot',
                'plot_shap_dependence_plot'
            ]
        if method_configs is None:
            method_configs = {}

        if os.path.exists(filename) and not overwrite:
            raise ValueError(f"File '{filename}' already exists. Set `overwrite=True` to replace it.")

        # Check that all methods called have been defined in this class
        for method_name in methods:
            if not hasattr(self, method_name):
                raise ValueError(f"Method '{method_name}' not found in {self.__class__.__name__}.")
            # Also check that the arguments are valid
            if method_name in method_configs:
                method = getattr(self, method_name)
                invalid_args = set(method_configs[method_name].keys()) - set(method.__code__.co_varnames)
                if invalid_args:
                    raise ValueError(f"Method '{method_name}' does not accept the following arguments: {invalid_args}")
                
        with PdfPages(filename) as pdf:
            for method_name in methods:

                plt.close('all')
                
                method = getattr(self, method_name)
                config = method_configs.get(method_name, {})
                
                # Force display=False in config if it's one of our plotting methods 
                if 'display' in method.__code__.co_varnames:
                    config['display'] = False

                # Call the method
                try:
                    method(**config)
                except TypeError:
                    print(f"{method_name} failed with config: {config}, skipping")
                    continue
                pdf.savefig(plt.gcf())
        
        print(f"Report saved to: {filename}")
    
