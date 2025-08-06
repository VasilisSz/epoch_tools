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


def permutation_cluster(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    subject_col: str,
    freq_col: str,
    node_cols: list[str],
    adjacency='auto',
    paired: bool = True,
    n_permutations: int = 1000,
    threshold: float | None = None,
    tail: int = 0,
    correction: str | None = 'fdr_bh',
    alpha: float = 0.05,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Cluster-based permutation test per node or edge, returns p-values per freq.

    Returns DataFrame with columns [*node_cols, freq_col, T_obs, p_uncorrected, p_corrected, significant]
    """
    # Unique keys for node/edge
    if len(node_cols) == 1:
        keys = df[node_cols[0]].unique()
    else:
        keys = df[node_cols].drop_duplicates().apply(tuple, axis=1)
    records = []
    for key in keys:
        # subset
        if len(node_cols) == 1:
            sub = df[df[node_cols[0]] == key]
        else:
            sub = df[(df[node_cols[0]] == key[0]) & (df[node_cols[1]] == key[1])]
        # pivot: subjects × freqs
        pivot = (
            sub
            .pivot_table(
                index=subject_col,
                columns=freq_col,
                values=value_col,
                aggfunc='mean'
            )
            .sort_index(axis=1)
        )
        freqs = pivot.columns.values
        data = pivot.values
        # split groups
        levels = sub[group_col].unique()
        if len(levels) != 2:
            raise ValueError("Need exactly two levels in group_col for permutation_cluster")
        g0, g1 = levels
        # matrices for each group
        mat0 = pivot.loc[sub[sub[group_col] == g0][subject_col].unique(), :].values
        mat1 = pivot.loc[sub[sub[group_col] == g1][subject_col].unique(), :].values
        # prepare adjacency
        use_adj = None
        if adjacency == 'auto' and len(node_cols) == 2:
            edge_df = sub[node_cols].drop_duplicates().reset_index(drop=True)
            use_adj = compute_edge_adjacency(edge_df[node_cols[0]], edge_df[node_cols[1]])
        elif isinstance(adjacency, sparse.spmatrix):
            use_adj = adjacency
        # run cluster test
        if paired:
            # align subjects
            common = np.intersect1d(
                pivot.index[pivot.index.isin(
                    sub[sub[group_col] == g0][subject_col]
                )],
                pivot.index[pivot.index.isin(
                    sub[sub[group_col] == g1][subject_col]
                )]
            )
            diff = (
                pivot.loc[common].loc[:, freqs].values
                - pivot.loc[common].loc[:, freqs].values
            )
            T_obs, clusters, p_vals, _ = mne_perm_cluster_1samp_test(
                X=diff,
                n_permutations=n_permutations,
                threshold=threshold,
                tail=tail,
                adjacency=use_adj,
                out_type='mask',
                seed=seed
            )
        else:
            T_obs, clusters, p_vals, _ = mne_perm_cluster_test(
                [mat0, mat1],
                n_permutations=n_permutations,
                threshold=threshold,
                tail=tail,
                adjacency=use_adj,
                out_type='mask',
                seed=seed
            )
        # normalize cluster representation
        # clusters: sequence of masks or index tuples
        masks = []
        n_times = len(freqs)
        for clu in clusters:
            if isinstance(clu, np.ndarray) and clu.dtype == bool:
                masks.append(clu)
            elif isinstance(clu, (tuple, list)):
                # index tuple: (_, time_inds)
                time_inds = clu[1] if len(clu) == 2 else clu[0]
                m = np.zeros(n_times, bool)
                m[time_inds] = True
                masks.append(m)
            else:
                raise ValueError("Unrecognized cluster format: %r" % (clu,))
        # collect per-frequency p_uncorrected
        for fi, f in enumerate(freqs):
            p_unc = 1.0
            for ci, mask in enumerate(masks):
                if mask[fi]:
                    p_unc = min(p_unc, p_vals[ci])
            rec = {freq_col: f, 'T_obs': T_obs[fi], 'p_uncorrected': p_unc}
            if len(node_cols) == 1:
                rec[node_cols[0]] = key
            else:
                rec[node_cols[0]] = key[0]
                rec[node_cols[1]] = key[1]
            records.append(rec)
    result = pd.DataFrame(records)
    # multiple comparisons
    if correction:
        rej, p_corr, _, _ = multipletests(result['p_uncorrected'], alpha=alpha, method=correction)
        result['p_corrected'] = p_corr
        result['significant'] = rej
    else:
        result['p_corrected'] = result['p_uncorrected']
        result['significant'] = result['p_uncorrected'] < alpha
    return result


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

