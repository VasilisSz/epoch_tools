import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
from mne.stats import (
    permutation_cluster_test as mne_perm_cluster_test,
    permutation_cluster_1samp_test as mne_perm_cluster_1samp_test
)
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def compute_edge_adjacency(node1, node2):
    """
    Build adjacency matrix for edges: edges are adjacent if they share a node.
    node1, node2 : array-like of same length, edge endpoints
    Returns
    -------
    adj : scipy.sparse.coo_matrix, shape (n_edges, n_edges)
        adjacency matrix (binary)
    """
    edges = list(zip(node1, node2))
    n = len(edges)
    mat = np.zeros((n, n), dtype=bool)
    for i, (a1, b1) in enumerate(edges):
        for j, (a2, b2) in enumerate(edges):
            if i >= j:
                continue
            if (a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2):
                mat[i, j] = mat[j, i] = True
    return sparse.coo_matrix(mat)

def permutation_test_cluster(
    df: pd.DataFrame,
    hue: str,
    value_col: str='con',
    subject_col: str = 'animal_id',
    freq_col: str = 'freq',
    node_cols: list[str] = ['node1', 'node2'],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    paired: bool = False,
    seed: int | None = None,
    threshold: float | None = None
) -> pd.DataFrame:
    """
    Cluster-based permutation test per node-pair for bivariate connectivity.
    
    Uses MNE's cluster-based permutation testing to identify clusters of 
    adjacent frequencies that show significant differences between groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with connectivity data
    hue : str
        Column defining groups to compare (must have exactly 2 levels)
    value_col : str
        Column containing connectivity values (e.g., 'con')
    subject_col : str
        Column identifying subjects/animals
    freq_col : str
        Column containing frequency values
    node_cols : list[str]
        Columns defining node pairs (e.g., ['node1', 'node2'])
    n_permutations : int
        Number of permutations
    alpha : float
        Significance threshold
    paired : bool
        Whether to use paired permutation test
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [node1, node2, freq, p_value, significant]
        where p_value is the cluster-corrected p-value for each frequency
    """
    from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
    
    group_col = hue
    levels = df[group_col].unique()
    if len(levels) != 2:
        raise ValueError("Need exactly two levels in group_col")
    g0, g1 = levels
    
    records = []
    
    # Get unique node pairs
    pairs = df[node_cols].drop_duplicates().itertuples(index=False)
    
    for node1, node2 in pairs:
        # Filter for this node pair
        pair_data = df[(df[node_cols[0]] == node1) & (df[node_cols[1]] == node2)]
        
        # Pivot to get subjects x frequencies for each group
        pivot = pair_data.pivot_table(
            index=subject_col, 
            columns=freq_col, 
            values=value_col, 
            aggfunc='mean'
        ).sort_index(axis=1)
        
        freqs = pivot.columns.values
        
        if len(freqs) == 0:
            continue
        
        # Split by group
        g0_subjects = pair_data[pair_data[group_col] == g0][subject_col].unique()
        g1_subjects = pair_data[pair_data[group_col] == g1][subject_col].unique()
        
        if paired:
            # Get common subjects for paired test
            common_subjects = np.intersect1d(g0_subjects, g1_subjects)
            if len(common_subjects) < 2:
                continue
            
            # Compute difference: group0 - group1 for each subject
            diff_data = pivot.loc[common_subjects].values  # This should be differences
            # Actually need to recompute to get proper differences
            pivot_g0 = pair_data[pair_data[group_col] == g0].pivot_table(
                index=subject_col, columns=freq_col, values=value_col, aggfunc='mean'
            ).loc[common_subjects].sort_index(axis=1)
            pivot_g1 = pair_data[pair_data[group_col] == g1].pivot_table(
                index=subject_col, columns=freq_col, values=value_col, aggfunc='mean'
            ).loc[common_subjects].sort_index(axis=1)
            
            diff_data = (pivot_g0 - pivot_g1).values  # shape: (n_subjects, n_freqs)
            
            # Run 1-sample cluster test on differences
            T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                X=diff_data,
                n_permutations=n_permutations,
                threshold=Nothresholdne,  # Use default t-threshold
                tail=0,  # Two-tailed
                out_type='mask',
                seed=seed
            )
        else:
            # Get data for each group
            mat0 = pivot.loc[pivot.index.isin(g0_subjects)].values
            mat1 = pivot.loc[pivot.index.isin(g1_subjects)].values
            
            if len(mat0) < 2 or len(mat1) < 2:
                continue
            
            # Run 2-sample cluster test
            T_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
                [mat0, mat1],
                n_permutations=n_permutations,
                threshold=threshold,  # Use default t-threshold
                tail=0,  # Two-tailed
                out_type='mask',
                seed=seed,
                verbose=False
            )
        
        # Map cluster p-values back to each frequency
        # Each frequency gets the minimum p-value of any cluster it belongs to
        freq_pvalues = np.ones(len(freqs))
        
        for cluster_mask, pval in zip(clusters, cluster_pv):
            if isinstance(cluster_mask, np.ndarray) and cluster_mask.dtype == bool:
                freq_pvalues[cluster_mask] = np.minimum(freq_pvalues[cluster_mask], pval)
            elif isinstance(cluster_mask, tuple):
                # Handle tuple format (from 1D clustering)
                time_inds = cluster_mask[0] if len(cluster_mask) == 1 else cluster_mask[1]
                freq_pvalues[time_inds] = np.minimum(freq_pvalues[time_inds], pval)
        
        # Store results
        for freq, p_val in zip(freqs, freq_pvalues):
            records.append({
                node_cols[0]: node1,
                node_cols[1]: node2,
                freq_col: freq,
                'p_value': p_val,
                'significant': p_val < alpha
            })
    
    return pd.DataFrame(records)


def permutation_test_pairwise(
    df: pd.DataFrame,
    hue: str,
    value_col: str='con',
    subject_col: str = 'animal_id',
    freq_col: str = 'freq',
    node_cols: list[str] = ['node1', 'node2'],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    paired: bool = False,
    correction: str | None = None,  # NEW: 'fdr_bh', 'bonferroni', or None
    correction_level: str = 'per_pair',  # NEW: 'per_pair' or 'global'
    seed: int | None = None
) -> pd.DataFrame:
    """
    Permutation test per node-pair and frequency for bivariate connectivity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with connectivity data
    hue : str
        Column defining groups to compare (must have exactly 2 levels)
    value_col : str
        Column containing connectivity values (e.g., 'con')
    subject_col : str
        Column identifying subjects/animals
    freq_col : str
        Column containing frequency values
    node_cols : list[str]
        Columns defining node pairs (e.g., ['node1', 'node2'])
    n_permutations : int
        Number of permutations
    alpha : float
        Significance threshold
    paired : bool
        Whether to use paired permutation test
    correction : str, optional
        Multiple comparison correction method:
        - None: no correction (most liberal)
        - 'fdr_bh': Benjamini-Hochberg FDR (recommended)
        - 'fdr_by': Benjamini-Yekutieli FDR (more conservative)
        - 'bonferroni': Bonferroni correction (most conservative)
    correction_level : str
        Level at which to apply correction:
        - 'per_pair': correct within each node pair separately (less conservative)
        - 'global': correct across all node pairs and frequencies (more conservative)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [node1, node2, freq, p_value, p_corrected, significant]
    """
    from statsmodels.stats.multitest import multipletests
    
    if seed is not None:
        np.random.seed(seed)
    
    group_col = hue
    levels = df[group_col].unique()
    if len(levels) != 2:
        raise ValueError("Need exactly two levels in group_col")
    g0, g1 = levels
    
    records = []
    
    # Get unique node pairs
    pairs = df[node_cols].drop_duplicates().itertuples(index=False)
    
    for node1, node2 in pairs:
        # Filter for this node pair
        pair_data = df[(df[node_cols[0]] == node1) & (df[node_cols[1]] == node2)]
        
        # Get unique frequencies
        freqs = sorted(pair_data[freq_col].unique())
        
        pair_records = []
        
        for freq in freqs:
            freq_data = pair_data[pair_data[freq_col] == freq]
            
            # Get data for each group
            g0_data = freq_data[freq_data[group_col] == g0]
            g1_data = freq_data[freq_data[group_col] == g1]
            
            if paired:
                # Merge on subject to ensure pairing
                merged = pd.merge(
                    g0_data[[subject_col, value_col]],
                    g1_data[[subject_col, value_col]],
                    on=subject_col,
                    suffixes=('_0', '_1')
                )
                
                if len(merged) == 0:
                    continue
                
                vals0 = merged[f"{value_col}_0"].values
                vals1 = merged[f"{value_col}_1"].values
                
                # Observed difference
                observed_diff = np.mean(vals0 - vals1)
                
                # Permutation test
                perm_diffs = []
                for _ in range(n_permutations):
                    # Random sign flips for paired data
                    signs = np.random.choice([-1, 1], size=len(vals0))
                    perm_diff = np.mean(signs * (vals0 - vals1))
                    perm_diffs.append(perm_diff)
                
                p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            
            else:
                vals0 = g0_data[value_col].values
                vals1 = g1_data[value_col].values
                
                if len(vals0) == 0 or len(vals1) == 0:
                    continue
                
                # Observed difference
                observed_diff = np.mean(vals0) - np.mean(vals1)
                
                # Permutation test
                combined = np.concatenate([vals0, vals1])
                n0, n1 = len(vals0), len(vals1)
                
                perm_diffs = []
                for _ in range(n_permutations):
                    np.random.shuffle(combined)
                    perm_diff = np.mean(combined[:n0]) - np.mean(combined[n0:])
                    perm_diffs.append(perm_diff)
                
                p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            
            rec = {
                node_cols[0]: node1,
                node_cols[1]: node2,
                freq_col: freq,
                'p_value': p_value
            }
            
            if correction_level == 'per_pair':
                pair_records.append(rec)
            else:
                records.append(rec)
        
        # Apply per-pair correction if requested
        if correction_level == 'per_pair' and correction is not None and len(pair_records) > 0:
            p_vals = [r['p_value'] for r in pair_records]
            reject, p_corr, _, _ = multipletests(p_vals, alpha=alpha, method=correction)
            
            for rec, p_c, sig in zip(pair_records, p_corr, reject):
                rec['p_corrected'] = p_c
                rec['significant'] = sig
            
            records.extend(pair_records)
        elif correction_level == 'per_pair':
            # No correction
            for rec in pair_records:
                rec['p_corrected'] = rec['p_value']
                rec['significant'] = rec['p_value'] < alpha
            records.extend(pair_records)
    
    # Apply global correction if requested
    if correction_level == 'global':
        if correction is not None and len(records) > 0:
            p_vals = [r['p_value'] for r in records]
            reject, p_corr, _, _ = multipletests(p_vals, alpha=alpha, method=correction)
            
            for rec, p_c, sig in zip(records, p_corr, reject):
                rec['p_corrected'] = p_c
                rec['significant'] = sig
        else:
            # No correction
            for rec in records:
                rec['p_corrected'] = rec['p_value']
                rec['significant'] = rec['p_value'] < alpha
    
    return pd.DataFrame(records)

# def permutation_test_pairwise(
#     df: pd.DataFrame,
#     hue: str,
#     value_col: str='con',
#     subject_col: str = 'animal_id',
#     freq_col: str = 'freq',
#     node_cols: list[str] = ['node1', 'node2'],
#     n_permutations: int = 1000,
#     alpha: float = 0.05,
#     paired: bool = False,
#     seed: int | None = None
# ) -> pd.DataFrame:
#     """
#     Permutation test per node-pair and frequency for bivariate connectivity.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame with connectivity data
#     hue : str
#         Column defining groups to compare (must have exactly 2 levels)
#     value_col : str
#         Column containing connectivity values (e.g., 'con')
#     subject_col : str
#         Column identifying subjects/animals
#     freq_col : str
#         Column containing frequency values
#     node_cols : list[str]
#         Columns defining node pairs (e.g., ['node1', 'node2'])
#     n_permutations : int
#         Number of permutations
#     alpha : float
#         Significance threshold
#     paired : bool
#         Whether to use paired permutation test
#     seed : int, optional
#         Random seed for reproducibility
        
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with columns [node1, node2, freq, p_value, significant]
#     """
#     if seed is not None:
#         np.random.seed(seed)
#     group_col = hue
#     levels = df[group_col].unique()
#     if len(levels) != 2:
#         raise ValueError("Need exactly two levels in group_col")
#     g0, g1 = levels
    
#     records = []
    
#     # Get unique node pairs
#     pairs = df[node_cols].drop_duplicates().itertuples(index=False)
    
#     for node1, node2 in pairs:
#         # Filter for this node pair
#         pair_data = df[(df[node_cols[0]] == node1) & (df[node_cols[1]] == node2)]
        
#         # Get unique frequencies
#         freqs = sorted(pair_data[freq_col].unique())
        
#         for freq in freqs:
#             freq_data = pair_data[pair_data[freq_col] == freq]
            
#             # Get data for each group
#             g0_data = freq_data[freq_data[group_col] == g0]
#             g1_data = freq_data[freq_data[group_col] == g1]
            
#             if paired:
#                 # Merge on subject to ensure pairing
#                 merged = pd.merge(
#                     g0_data[[subject_col, value_col]],
#                     g1_data[[subject_col, value_col]],
#                     on=subject_col,
#                     suffixes=('_0', '_1')
#                 )
                
#                 if len(merged) == 0:
#                     continue
                
#                 vals0 = merged[f"{value_col}_0"].values
#                 vals1 = merged[f"{value_col}_1"].values
                
#                 # Observed difference
#                 observed_diff = np.mean(vals0 - vals1)
                
#                 # Permutation test
#                 perm_diffs = []
#                 for _ in range(n_permutations):
#                     # Random sign flips for paired data
#                     signs = np.random.choice([-1, 1], size=len(vals0))
#                     perm_diff = np.mean(signs * (vals0 - vals1))
#                     perm_diffs.append(perm_diff)
                
#                 p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            
#             else:
#                 vals0 = g0_data[value_col].values
#                 vals1 = g1_data[value_col].values
                
#                 if len(vals0) == 0 or len(vals1) == 0:
#                     continue
                
#                 # Observed difference
#                 observed_diff = np.mean(vals0) - np.mean(vals1)
                
#                 # Permutation test
#                 combined = np.concatenate([vals0, vals1])
#                 n0, n1 = len(vals0), len(vals1)
                
#                 perm_diffs = []
#                 for _ in range(n_permutations):
#                     np.random.shuffle(combined)
#                     perm_diff = np.mean(combined[:n0]) - np.mean(combined[n0:])
#                     perm_diffs.append(perm_diff)
                
#                 p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            
#             records.append({
#                 node_cols[0]: node1,
#                 node_cols[1]: node2,
#                 freq_col: freq,
#                 'p_value': p_value,
#                 'significant': p_value < alpha
#             })
    
#     return pd.DataFrame(records)

# def permutation_cluster(
#     df: pd.DataFrame,
#     value_col: str,
#     group_col: str,
#     subject_col: str,
#     freq_col: str,
#     node_cols: list[str],
#     adjacency='auto',
#     paired: bool = True,
#     n_permutations: int = 1000,
#     threshold: float | None = None,
#     tail: int = 0,
#     correction: str | None = 'fdr_bh',
#     alpha: float = 0.05,
#     seed: int | None = None
# ) -> pd.DataFrame:
#     """
#     Cluster-based permutation test per node or edge, returns p-values per freq.

#     Returns DataFrame with columns [*node_cols, freq_col, T_obs, p_uncorrected, p_corrected, significant]
#     """
#     # Unique keys for node/edge
#     if len(node_cols) == 1:
#         keys = df[node_cols[0]].unique()
#     else:
#         keys = df[node_cols].drop_duplicates().apply(tuple, axis=1)
#     records = []
#     for key in keys:
#         # subset
#         if len(node_cols) == 1:
#             sub = df[df[node_cols[0]] == key]
#         else:
#             sub = df[(df[node_cols[0]] == key[0]) & (df[node_cols[1]] == key[1])]
#         # pivot: subjects × freqs
#         pivot = (
#             sub
#             .pivot_table(
#                 index=subject_col,
#                 columns=freq_col,
#                 values=value_col,
#                 aggfunc='mean'
#             )
#             .sort_index(axis=1)
#         )
#         freqs = pivot.columns.values
#         data = pivot.values
#         # split groups
#         levels = sub[group_col].unique()
#         if len(levels) != 2:
#             raise ValueError("Need exactly two levels in group_col for permutation_cluster")
#         g0, g1 = levels
#         # matrices for each group
#         mat0 = pivot.loc[sub[sub[group_col] == g0][subject_col].unique(), :].values
#         mat1 = pivot.loc[sub[sub[group_col] == g1][subject_col].unique(), :].values
#         # prepare adjacency
#         use_adj = None
#         if adjacency == 'auto' and len(node_cols) == 2:
#             edge_df = sub[node_cols].drop_duplicates().reset_index(drop=True)
#             use_adj = compute_edge_adjacency(edge_df[node_cols[0]], edge_df[node_cols[1]])
#         elif isinstance(adjacency, sparse.spmatrix):
#             use_adj = adjacency
#         # run cluster test
#         if paired:
#             # align subjects
#             common = np.intersect1d(
#                 pivot.index[pivot.index.isin(
#                     sub[sub[group_col] == g0][subject_col]
#                 )],
#                 pivot.index[pivot.index.isin(
#                     sub[sub[group_col] == g1][subject_col]
#                 )]
#             )
#             diff = (
#                 pivot.loc[common].loc[:, freqs].values
#                 - pivot.loc[common].loc[:, freqs].values
#             )
#             T_obs, clusters, p_vals, _ = mne_perm_cluster_1samp_test(
#                 X=diff,
#                 n_permutations=n_permutations,
#                 threshold=threshold,
#                 tail=tail,
#                 adjacency=use_adj,
#                 out_type='mask',
#                 seed=seed
#             )
#         else:
#             T_obs, clusters, p_vals, _ = mne_perm_cluster_test(
#                 [mat0, mat1],
#                 n_permutations=n_permutations,
#                 threshold=threshold,
#                 tail=tail,
#                 adjacency=use_adj,
#                 out_type='mask',
#                 seed=seed
#             )
#         # normalize cluster representation
#         # clusters: sequence of masks or index tuples
#         masks = []
#         n_times = len(freqs)
#         for clu in clusters:
#             if isinstance(clu, np.ndarray) and clu.dtype == bool:
#                 masks.append(clu)
#             elif isinstance(clu, (tuple, list)):
#                 # index tuple: (_, time_inds)
#                 time_inds = clu[1] if len(clu) == 2 else clu[0]
#                 m = np.zeros(n_times, bool)
#                 m[time_inds] = True
#                 masks.append(m)
#             else:
#                 raise ValueError("Unrecognized cluster format: %r" % (clu,))
#         # collect per-frequency p_uncorrected
#         for fi, f in enumerate(freqs):
#             p_unc = 1.0
#             for ci, mask in enumerate(masks):
#                 if mask[fi]:
#                     p_unc = min(p_unc, p_vals[ci])
#             rec = {freq_col: f, 'T_obs': T_obs[fi], 'p_uncorrected': p_unc}
#             if len(node_cols) == 1:
#                 rec[node_cols[0]] = key
#             else:
#                 rec[node_cols[0]] = key[0]
#                 rec[node_cols[1]] = key[1]
#             records.append(rec)
#     result = pd.DataFrame(records)
#     # multiple comparisons
#     if correction:
#         rej, p_corr, _, _ = multipletests(result['p_uncorrected'], alpha=alpha, method=correction)
#         result['p_corrected'] = p_corr
#         result['significant'] = rej
#     else:
#         result['p_corrected'] = result['p_uncorrected']
#         result['significant'] = result['p_uncorrected'] < alpha
#     return result


def mass_univariate_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    subject_col: str | None,
    factor_col: str,
    node_cols: list[str],
    paired: bool = True,
    test: str = 'parametric',
    correction: str | None = 'fdr_bh',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    One test per node/edge × factor, with global correction.

    Returns columns [*node_cols, factor, stat, p_uncorrected, p_corrected, significant]
    """
    levels = df[group_col].unique()
    if len(levels) != 2:
        raise ValueError("Exactly two groups required for mass_univariate_test")
    g0, g1 = levels
    records = []
    if len(node_cols) == 1:
        keys = df[node_cols[0]].unique()
    else:
        keys = df[node_cols].drop_duplicates().apply(tuple, axis=1)
    for key in keys:
        if len(node_cols) == 1:
            sub = df[df[node_cols[0]] == key]
        else:
            sub = df[(df[node_cols[0]] == key[0]) & (df[node_cols[1]] == key[1])]
        for lvl, grp in sub.groupby(factor_col, sort=False):
            a = grp[grp[group_col] == g0][value_col].values
            b = grp[grp[group_col] == g1][value_col].values
            if paired:
                if subject_col is None:
                    raise ValueError("paired=True requires subject_col")
                merged = pd.merge(
                    grp[[subject_col, value_col]][grp[group_col] == g0],
                    grp[[subject_col, value_col]][grp[group_col] == g1],
                    on=subject_col,
                    suffixes=('_0','_1')
                )
                x = merged[f"{value_col}_0"].values
                y = merged[f"{value_col}_1"].values
            else:
                x, y = a, b
            if test == 'parametric':
                stat, p = (stats.ttest_rel if paired else stats.ttest_ind)(x, y, nan_policy='omit')
            elif test == 'rank':
                stat, p = (stats.wilcoxon if paired else stats.mannwhitneyu)(x, y)
            else:
                raise ValueError("test must be 'parametric' or 'rank'")
            rec = {factor_col: lvl, 'stat': stat, 'p_uncorrected': p}
            if len(node_cols) == 1:
                rec[node_cols[0]] = key
            else:
                rec[node_cols[0]] = key[0]
                rec[node_cols[1]] = key[1]
            records.append(rec)
    res = pd.DataFrame(records)
    if correction:
        rej, p_corr, _, _ = multipletests(res['p_uncorrected'], alpha=alpha, method=correction)
        res['p_corrected'] = p_corr
        res['significant'] = rej
    else:
        res['p_corrected'] = res['p_uncorrected']
        res['significant'] = res['p_uncorrected'] < alpha
    return res


def run_mixedlm(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    subject_col: str,
    node_cols: list[str],
    additional_fixed: list[str] | None = None
) -> pd.DataFrame:
    """
    Fit mixed model per node/edge: value ~ C(group) + additional_fixed + (1|subject)

    Returns concatenated summary DataFrame.
    """
    summaries = []
    if len(node_cols) == 1:
        keys = df[node_cols[0]].unique()
    else:
        keys = df[node_cols].drop_duplicates().apply(tuple, axis=1)
    for key in keys:
        sub = (df[df[node_cols[0]] == key] if len(node_cols)==1 else
               df[(df[node_cols[0]]==key[0]) & (df[node_cols[1]]==key[1])])
        fixed = [f"C({group_col})"] + (additional_fixed or [])
        formula = f"{value_col} ~ " + " + ".join(fixed)
        md = mixedlm(formula, sub, groups=sub[subject_col])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            mdf = md.fit(reml=False)
        summ = mdf.summary().tables[1]
        df_summ = pd.DataFrame(summ)
        for col, val in zip(node_cols, key if len(node_cols)>1 else [key]):
            df_summ[col] = val
        summaries.append(df_summ)
    return pd.concat(summaries, ignore_index=True)

