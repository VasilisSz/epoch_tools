"""
A wrapper for the `connectivity` analysis using mne_connectivity.
The majority of the methods are called in et.Epochs.compare_con

"""

import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy import signal, stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx

import re



def compute_con(epochs, method, fmin=0, fmax=100, **kwargs):
    """
        Wrapper to compute spectral connectivity for MNE Epochs objects.

        Parameters:
        epochs : mne.Epochs
            The MNE Epochs object containing the EEG data.
        method : str
            Connectivity measure(s) to compute. Calls mne_connectivity.spectral_connectivity_epochs
            These can be ['coh', 'cohy', 'imcoh', 'mic', 'mim', 'plv', 
            'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased', 'gc', 'gc_tr']. 
            Multivariate methods (['mic', 'mim', 'gc', 'gc_tr]) cannot be called with the other methods.
        fmin : float, optional
            The minimum frequency of interest (default is 0).
        fmax : float, optional
            The maximum frequency of interest (default is 100).
        *kwargs : dict, optional
            Additional arguments passed to the spectral_connectivity_epochs function.

        Returns:
        numpy.ndarray
            The connectivity matrix.
    """
    con = spectral_connectivity_epochs(
        epochs, method=method, mode='multitaper', sfreq=epochs.info['sfreq'],
        fmin=fmin, fmax=fmax, faverage=True, verbose="ERROR", gc_n_lags=40, **kwargs)
    return con.get_data(output='dense')


def connectivity_df(epochs, method, grouper, per_subject=True, freq_bands = None):
    """
    Compute connectivity for each subject or group and organize the results in a DataFrame.

    Parameters:
    epochs : mne.Epochs
        The MNE Epochs object containing the EEG data.
    method : str
        The method to use for computing connectivity (e.g., 'wpli2_debiased', 'dpli', 'coh,  etc.).
    per_subject : bool, optional
        If True, compute connectivity per subject. If False, compute connectivity per genotype (default is True).
    freq_bands : dict, optional
        Dictionary of frequency bands with their corresponding (fmin, fmax) tuples 
        (default is None, which uses standard frequency bands).

    Returns:
    pandas.DataFrame
        DataFrame containing the connectivity results with columns for 
        connectivity value, nodes, frequency band, subject/genotype.
    """
    if freq_bands is None:
        freq_bands = {'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}

    metadata = epochs.metadata
    groups = metadata[grouper].unique()
    # DataFrame to store the results
    results_df = pd.DataFrame()

    if per_subject:
        mouse_ids = metadata['animal_id'].unique()

        # Iterate over epochs
        for mouse_id in mouse_ids:
            # Subset data
            mouse_epochs = epochs[metadata['animal_id']==mouse_id].copy()
            mouse_group = mouse_epochs.metadata[grouper].unique()[0]
            # print(genotype)
        
            # Compute connectivity for each frequency band
            for band_name, (fmin, fmax) in freq_bands.items():
                con = compute_con(mouse_epochs, method, fmin, fmax)

                # Flatten connectivity matrix and add to the dictionary with appropriate names
                for i in range(con.shape[1]):
                    for j in range(i+1, con.shape[0]):
                        results_df = pd.concat([results_df, pd.DataFrame(
                            {
                                'con' : con[j, i, 0],
                                'node1' : epochs.ch_names[i],
                                'node2' : epochs.ch_names[j],
                                'band' : band_name,
                                'mouse_id': mouse_id,
                                'group':mouse_group
                            }, index=[0]
                        )], ignore_index=True)

        return results_df
    else:
        for group in groups:
            genotype_epochs = epochs[epochs.metadata[grouper]==group].copy()
            for band_name, (fmin, fmax) in freq_bands.items():
                con = compute_con(genotype_epochs, method, fmin, fmax)
                for i in range(con.shape[1]):
                    for j in range(i+1, con.shape[0]):
                        results_df = pd.concat([results_df, pd.DataFrame(
                            {
                                'con' : con[j, i, 0],
                                'node1' : epochs.ch_names[i],
                                'node2' : epochs.ch_names[j],
                                'band' : band_name,
                                'genotype':group
                            }, index=[0]
                        )], ignore_index=True)
        return results_df
    
def plot_con_heatmap(con_df, freq_band, con_col = 'con', vmin=0, vmax=1, fig_title = '', 
                     method='', cmap='viridis', ax=None):
    """
    Generate a heatmap of mean connectivity values for a specific frequency band. Used on
    an existing Axes.

    Parameters:
    con_df : pandas.DataFrame
        The DataFrame containing the connectivity data.
    freq_band : str
        The frequency band of interest.
    con_col : str, optional
        The column name for connectivity values in the DataFrame (default is 'con').
    vmin : float, optional
        The minimum value for the colormap (default is 0).
    vmax : float, optional
        The maximum value for the colormap (default is 1).
    fig_title : str, optional
        The title of the figure (default is '').
    method : str, optional
        The connectivity method used (default is '').
    cmap : str, optional
        The colormap to use for the heatmap (default is 'viridis').
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the heatmap (default is None).

    Returns: Nothing
    """
    # Filter data for the specified frequency band
    filtered_df = con_df[con_df['band'] == freq_band]

    # Create a pivot table with mean wPLI values
    pivot_table = filtered_df.pivot_table(index='node1', columns='node2', values=con_col, aggfunc='mean')

    # Ensure the table is symmetric by filling NaN values
    pivot_table = pivot_table.combine_first(pivot_table.T)

    # Set the diagonal and upper triangle values to NaN
    for i in range(pivot_table.shape[0]):
        for j in range(i, pivot_table.shape[1]):
            pivot_table.iat[i, j] = np.nan

    # Generate heatmap
    fig = sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt=".2f",  vmin=vmin, vmax=vmax, 
                      ax=ax, square=True, cbar_kws={'shrink': 0.75})
    fig.set_title(fig_title)

    cbar = fig.collections[0].colorbar
    cbar.set_label(f'Mean {method}', fontsize=12)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

def plot_con_barplots(con_df, method, grouper, hue_order):
    """
        Generate bar plots of connectivity values for each node pair across different frequency bands.

        Parameters:
        con_df (DataFrame): DataFrame containing the connectivity data.
        method (str): The connectivity method used (e.g., 'pli', 'wpli').

        Returns:
        None: This function generates and displays a series of bar plots.
    """
    combinations = con_df['node1'] + '-' + con_df['node2']
    unique_combinations = combinations.unique()
    
    fig, axs = plt.subplots(nrows=3, ncols=7, figsize=(35, 15), sharey=True, sharex=True)
    axs = axs.ravel()
    
    for i, combination in enumerate(unique_combinations):
        # subset the data
        chan1, chan2 = combination.split('-')
        data_subset = con_df[(con_df['node1'] == chan1) & (con_df['node2'] == chan2)]
        
        sns.barplot(data=data_subset, x="band", y="con", hue=grouper, hue_order=hue_order, ax=axs[i])
        sns.stripplot(data=data_subset, x="band", y="con", hue=grouper, palette='dark:black', 
                      hue_order=hue_order, alpha=0.4, dodge=True, legend=False, ax=axs[i])
        
        axs[i].set_title(f"Connectivity of {chan1} and {chan2} ({method.upper()})")
        axs[i].set_xlabel("Frequency band")
        axs[i].set_ylabel(f"Connectivity ({method.upper()})")
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()

def plot_con_boxplots(con_df, method, grouper, hue_order, stat_results=None):
    """
        Generate box plots of connectivity values for each node pair across different 
            frequency bands, with optional statistical significance markers.

        Parameters:
        con_df (DataFrame): DataFrame containing the connectivity data.
        method (str): The connectivity method used (e.g., 'pli', 'wpli').
        stat_results (DataFrame, optional): DataFrame containing statistical results, 
            with columns ['Node1', 'Node2', 'Band', 'P-Value'].

        Returns:
        None: This function generates and displays a series of box plots.
    """

    combinations = con_df['node1'] + '-' + con_df['node2']
    unique_combinations = combinations.unique()
    
    fig, axs = plt.subplots(nrows=3, ncols=7, figsize=(35, 15), sharey=True, sharex=True)
    axs = axs.ravel()
    
    for i, combination in enumerate(unique_combinations):
        # subset the data
        chan1, chan2 = combination.split('-')
        data_subset = con_df[(con_df['node1'] == chan1) & (con_df['node2'] == chan2)]
        
        sns.boxplot(data=data_subset, x="band", y="con", hue=grouper, hue_order=hue_order, ax=axs[i])
        sns.stripplot(data=data_subset, x="band", y="con", hue=grouper, palette='dark:black', 
                      hue_order=hue_order, alpha=0.4, dodge=True, legend=False, ax=axs[i])
        
        axs[i].set_title(f"Connectivity of {chan1} and {chan2} ({method.upper()})")
        axs[i].set_xlabel("Frequency band")
        axs[i].set_ylabel(f"Connectivity ({method.upper()})")
        axs[i].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[i].yaxis.set_tick_params(which='both', labelleft=True)

        # Add significance stars if star_results is provided
        if stat_results is not None:
            p_val = stat_results[(stat_results['Node1']==chan1) & (stat_results['Node2']==chan2)][['Band','P-Value']]
            for k, band in enumerate(p_val['Band'].unique()):
                if p_val[p_val['Band']==band]['P-Value'].values[0] < 0.05:
                    x_pos = (k+1)*0.2
                    axs[i].text(band, 1, "*", horizontalalignment='center', verticalalignment='center', 
                                fontsize=20, color='red')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()

def plot_group_heatmaps(data, method, grouper, cmap='flare', **kwargs):

    """
        Generate heatmaps of connectivity values for different genotypes across frequency bands,
          with an optional difference plot.

        Parameters:
        data (DataFrame): DataFrame containing the connectivity data.
        method (str): The connectivity method used (e.g., 'pli', 'wpli').
        cmap (str, optional): Colormap for the heatmaps. Default is 'flare'.
        plot_diff (bool, optional): If True, plot the percentage difference between genotypes. Default is False.
        **kwargs: Additional keyword arguments for the heatmap plotting function.

        Returns:
        None: This function generates and displays a series of heatmaps.
    """
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    
    freq_bands = {'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}
    freq_band_names = freq_bands.keys()
    groups = data[grouper].unique()
    
    # loop through genotypes
    for i, group in enumerate(groups):
        group_data = data[data[grouper] == group]
        for j, band in enumerate(freq_band_names):
            plot_con_heatmap(group_data, band, cmap=cmap, fig_title=f'{group} - {band} Band', 
                                method=method, ax=ax[i, j])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=-0.1)

def plot_genotype_heatmaps(data, method, cmap='flare', plot_diff = False, **kwargs):

    """
        Generate heatmaps of connectivity values for different genotypes across frequency bands,
          with an optional difference plot.

        Parameters:
        data (DataFrame): DataFrame containing the connectivity data.
        method (str): The connectivity method used (e.g., 'pli', 'wpli').
        cmap (str, optional): Colormap for the heatmaps. Default is 'flare'.
        plot_diff (bool, optional): If True, plot the percentage difference between genotypes. Default is False.
        **kwargs: Additional keyword arguments for the heatmap plotting function.

        Returns:
        None: This function generates and displays a series of heatmaps.
    """
    if plot_diff:
        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(24, 15))
        
        freq_bands = {'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}
        freq_band_names = freq_bands.keys()
        genotypes = data['genotype'].unique()
        
        # loop through genotypes
        for i, genotype in enumerate(genotypes):
            genotype_data = data[data['genotype'] == genotype]
            for j, band in enumerate(freq_band_names):
                plot_con_heatmap(genotype_data, band, cmap=cmap, fig_title=f'{genotype} - {band} Band', 
                                 method=method, ax=ax[i, j])
        
        # Plot difference heatmaps
        for j, band in enumerate(freq_band_names):
            # Caclulate con difference
            band_data = data[data['band']==band].groupby(['node1', 'node2','genotype'], as_index=False)['con'].mean()
            band_data = band_data.pivot_table(index=['node1', 'node2'], columns='genotype', values='con').reset_index()
            band_data['delta_con'] = (band_data['DRD2-KO'] - band_data['DRD2-WT'])/band_data['DRD2-WT'] * 100
            band_data['band'] = band
            plot_con_heatmap(band_data, band, con_col='delta_con', cmap='coolwarm', 
                             fig_title=f'% {method} Difference, {band} Band', method=method, ax=ax[-1, j], **kwargs)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=-0.1)
    else:
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
        
        freq_bands = {'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}
        freq_band_names = freq_bands.keys()
        genotypes = data['genotype'].unique()
        
        # loop through genotypes
        for i, genotype in enumerate(genotypes):
            genotype_data = data[data['genotype'] == genotype]
            for j, band in enumerate(freq_band_names):
                plot_con_heatmap(genotype_data, band, cmap=cmap, fig_title=f'{genotype} - {band} Band', 
                                 method=method, ax=ax[i, j])
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=-0.1)

def plot_delta_heatmaps(data, method, cmap='flare', **kwargs):
    """
        Generate heatmaps of percentage connectivity differences between genotypes across frequency bands.

        Parameters:
        data (DataFrame): DataFrame containing the connectivity data.
        method (str): The connectivity method used (e.g., 'pli', 'wpli').
        cmap (str, optional): Colormap for the heatmaps. Default is 'flare'.
        **kwargs: Additional keyword arguments for the heatmap plotting function.

        Returns:
        None: This function generates and displays a series of heatmaps showing percentage differences.
    """
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    
    freq_bands = {'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}
    freq_band_names = freq_bands.keys()

    for j, band in enumerate(freq_band_names):
        # Caclulate con difference
        band_data = data[data['band']==band].groupby(['node1', 'node2','genotype'], as_index=False)['con'].mean()
        band_data = band_data.pivot_table(index=['node1', 'node2'], columns='genotype', values='con').reset_index()
        band_data['delta_con'] = (band_data['DRD2-KO'] - band_data['DRD2-WT'])/band_data['DRD2-WT'] * 100
        band_data['band'] = band
        plot_con_heatmap(band_data, band, con_col='delta_con', cmap=cmap, 
                         fig_title=f'% {method} Difference, {band} Band', method=method, ax=ax[j], **kwargs)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=-0.1)

# Function to create and plot network graph
def plot_network(fig, ax, df, positions, title, con_col='con',  cmap='viridis', label='Connectivity'):
    """
    Create and plot a network graph based on connectivity data.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object to plot on.
    - df (pandas.DataFrame): DataFrame containing connectivity data 
        with columns 'node1', 'node2', and the connectivity column (default 'con').
    - positions (dict): Dictionary with node positions.
    - title (str): Title of the plot.
    - con_col (str, optional): Column name in df representing connectivity values. Default is 'con'.
    - cmap (str, optional): Colormap for the edges. Default is 'viridis'.
    - label (str, optional): Label for the colorbar. Default is 'Connectivity'.

    Returns:
    - sm (matplotlib.cm.ScalarMappable): ScalarMappable object for colorbar.
    """
    # Normalize the connectivity values for coloring
    if con_col == 'delta_con':
        norm = mcolors.Normalize(vmin=-50, vmax=50)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(cmap)
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['node1'], row['node2'], weight=row[con_col])
    
    # Get edge colors based on connectivity values
    edge_colors = [cmap(norm(row[con_col])) for _, row in df.iterrows()]
    
    edge_widths = [norm(row[con_col]) * 5 for _, row in df.iterrows()]  # Multiply by a factor to adjust thickness
    
    nx.draw(G, positions, ax=ax, with_labels=True, node_size=700, node_color='lightblue', 
            font_size=10, font_weight='bold', edge_color=edge_colors, width=edge_widths)
    # nx.draw_networkx_edge_labels(G, nx.spring_layout(G),ax=ax,  edge_labels=nx.get_edge_attributes(G, 'weight'))
    # nx.draw(G, positions, ax=ax, with_labels=True, node_size=700, node_color='lightblue', 
    #         font_size=10, font_weight='bold', edge_color=edge_colors)
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Connectivity')
    return sm

# Function to generate grid plot for WT vs KO
def plot_genotype_networks(data, method, coords, cmap='viridis'):
    """
    Generate grid plot comparing WT and KO genotypes across frequency bands.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing connectivity data 
        with columns 'genotype', 'band', 'node1', 'node2', and 'con'.
    - method (str): Method of connectivity calculation.
    - coords (dict): Dictionary with node positions.
    - cmap (str, optional): Colormap for the edges. Default is 'viridis'.

    Returns:
    - None
    """
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(28, 10))
    
    freq_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    genotypes = data['genotype'].unique()
    
    for i, genotype in enumerate(genotypes):
        genotype_data = data[data['genotype'] == genotype]
        for j, band in enumerate(freq_bands):
            band_data = genotype_data[genotype_data['band'] == band]
            avg_data = band_data.groupby(['node1', 'node2'], as_index=False)['con'].mean()
            plot_network(fig, ax[i, j], avg_data, coords, f'{genotype} - {band}', cmap=cmap, label=method)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.4)

# Function to generate grid plot for WT vs KO
def plot_delta_networks(data, method, coords, cmap='coolwarm'):
    """
    Generate grid plot comparing connectivity differences between WT and KO genotypes across frequency bands.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing connectivity data 
        with columns 'band', 'node1', 'node2', 'genotype', and 'con'.
    - method (str): Method of connectivity calculation.
    - coords (dict): Dictionary with node positions.
    - cmap (str, optional): Colormap for the edges. Default is 'coolwarm'.

    Returns:
    - None
    """
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(28, 5))
    
    freq_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    for j, band in enumerate(freq_bands):
        # Caclulate con difference
        band_data = data[data['band']==band].groupby(['node1', 'node2','genotype'], as_index=False)['con'].mean()
        band_data = band_data.pivot_table(index=['node1', 'node2'], columns='genotype', values='con').reset_index()
        band_data['delta_con'] = (band_data['DRD2-KO'] - band_data['DRD2-WT'])/band_data['DRD2-WT'] * 100

        plot_network(fig, ax[j], band_data, coords, f' KO - WT- {band}', con_col='delta_con', cmap=cmap, label=method)
        band_data['band'] = band
        all_data = pd.concat([all_data, band_data], ignore_index=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.4)

def compute_multi_con(data, method, mean_subj = True, **kwargs):
    """
    Compute multi-channel connectivity measures.

    Parameters:
    - data (mne.Epochs): MNE Epochs object containing the data.
    - method (str): Method of connectivity calculation.
    - mean_subj (bool, optional): Whether to average connectivity measures for each subject. Default is True.
    - kwargs: Additional keyword arguments for the connectivity calculation method.

    Returns:
    - results_df (pandas.DataFrame): DataFrame containing connectivity results 
        with columns 'animal_id', 'genotype', 'freqs', and 'con'.
    """
    results_df = pd.DataFrame()

    if mean_subj:
        subjects = data.metadata['animal_id'].unique()

        for subject in subjects:
            subject_data = data[data.metadata['animal_id'] == subject]
            genotype = subject_data.metadata['genotype'].unique()[0]

            con = spectral_connectivity_epochs(subject_data, method=method, n_jobs=-1, 
                                               fmin=1, fmax=100, verbose='WARNING', **kwargs)

            results_df = pd.concat([results_df, pd.DataFrame({
                'animal_id' : subject,
                'genotype' : genotype,
                'freqs' : con.freqs,
                'con' : con.get_data()[0,:]
            })], ignore_index=True)

        return results_df
    else:
        genotypes = data.metadata['genotype'].unique()

        for genotype in genotypes:
            genotype_data = data[data.metadata['genotype'] == genotype]
            con = spectral_connectivity_epochs(genotype_data, method=method, n_jobs=-1, 
                                               fmin=1, fmax=100, verbose='WARNING', **kwargs)

            results_df = pd.concat([results_df, pd.DataFrame({
                'genotype' : genotype,
                'freqs' : con.freqs,
                'con' : con.get_data()[0,:]
            })], ignore_index=True)

        return results_df
    
def con_stats(df, grouper, glabel1, glabel2, test_type='anova', correction=None, only_significant=False, alpha=0.05):
    """
    Perform statistical analysis on connectivity data.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing connectivity data.
    - test_type (str, optional): The type of statistical test to perform. Default is 'anova'.
        - 'anova': One-way ANOVA
        - 'ttest': Independent t-test
        - 'mannwhitney': Mann-Whitney U test
        - 'ranksums': Wilcoxon rank-sum test
    - correction (str, optional): The method for multiple comparison correction. Default is None.
        - None: No correction applied
        - 'bonferroni': Bonferroni correction
        - 'holm': Holm-Bonferroni correction
        - 'fdr_bh': Benjamini-Hochberg correction
    - only_significant (bool, optional): Whether to include only significant results. Default is False.
    - alpha (float, optional): The significance level for hypothesis testing. Default is 0.05.

    Returns:
    - results (pandas.DataFrame): The results of the statistical analysis, including band, node pair,
      test statistic, p-value, and corrected p-value (if correction is applied).

    Raises:
    - ValueError: If an invalid test type is specified.

    """
    from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, ranksums
    from statsmodels.stats.multitest import multipletests
    
    node_pairs = df[['node1', 'node2']].drop_duplicates()
    results = []
    
    # Iterate over each frequency band and node pair
    for band in df['band'].unique():
        for _, row in node_pairs.iterrows():
            node1, node2 = row['node1'], row['node2']
            
            # Filter the data for the current band and node pair
            subset = df[(df['band'] == band) & (df['node1'] == node1) & (df['node2'] == node2)]
            
            # Separate the data for WT and KO mice
            wt_data = subset[subset[grouper] == glabel1]['con']
            ko_data = subset[subset[grouper] == glabel2]['con']
            
            # Ensure there is data for both genotypes
            if len(wt_data) > 0 and len(ko_data) > 0:
                if test_type == 'anova':
                    # Perform one-way ANOVA
                    stat, p_value = f_oneway(wt_data, ko_data)
                elif test_type == 'ttest':
                    # Perform independent t-test
                    stat, p_value = ttest_ind(wt_data, ko_data)
                elif test_type == 'mannwhitney':
                    # Perform Mann-Whitney U test
                    stat, p_value = mannwhitneyu(wt_data, ko_data)
                elif test_type == 'ranksums':
                    # Perform Wilcoxon rank-sum test
                    stat, p_value = ranksums(wt_data, ko_data)
                else:
                    raise ValueError("Invalid test type specified. Use 'anova', 'ttest', 'mannwhitney', or 'ranksums'.")
                
                results.append([band, node1, node2, stat, p_value])
            else:
                # If data is missing for either genotype, append NaNs
                raise ValueError("Something is off ")
                # results.append([band, node1, node2, float('nan'), float('nan')])

    # Convert the results into a DataFrame
    results = pd.DataFrame(results, columns=['Band', 'Node1', 'Node2', 'Statistic', 'P-Value'])

    if correction:
        # Apply multiple comparison correction
        corrected_p_values = multipletests(results['P-Value'], alpha=alpha, method=correction)[1]
        results['Corrected P-Value'] = corrected_p_values

        if only_significant:
            results = results[results['Corrected P-Value'] < alpha]
    else:
        if only_significant:
            results = results[results['P-Value'] < alpha]

    return results

