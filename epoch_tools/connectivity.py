"""
A wrapper for the `connectivity` analysis using mne_connectivity.
The majority of the methods are called in et.Epochs.compare_con

"""

import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy import signal, stats

from .utils import row_col_layout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx

import re

from tqdm import tqdm


def compute_con(epochs, method, fmin=1, fmax=100, **kwargs):
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

def connectivity_df(
    epochs: mne.Epochs,
    method: str,
    hue,                        # str or list of str
    *,
    avg_level: str = "all",
    freq_bands: dict | None = None,
):

    if freq_bands is None:
        freq_bands = {
            "Delta": (2, 4), "Theta": (4, 8),
            "Alpha": (8,13), "Beta": (13,30), "Gamma": (30,100)
        }

    # normalize hue_cols
    hue_cols = [hue] if isinstance(hue, str) else list(hue)
    for c in hue_cols:
        if c not in epochs.metadata.columns:
            raise ValueError(f"Hue column {c!r} not found")

    # decide which columns to group by
    if avg_level == "subject":
        if "animal_id" not in epochs.metadata.columns:
            raise ValueError("avg_level='subject' needs 'animal_id' in metadata")
        group_cols = ["animal_id"] + hue_cols
    elif avg_level == "all":
        group_cols = hue_cols
    else:
        raise ValueError("avg_level must be 'all' or 'subject'")

    # build results
    rows = []
    chs  = epochs.ch_names

    # loop over every unique combination of group_cols
    for combo, submeta in tqdm(epochs.metadata.groupby(group_cols), desc="Computing connectivity"):
        # `combo` is a tuple of values (or a single value if len=1)
        # subset the epochs
        mask  = np.all([epochs.metadata[col] == submeta.iloc[0][col]
                        for col in group_cols], axis=0)
        subepo = epochs[mask]

        # compute connectivity once on that subset
        for band, (fmin, fmax) in freq_bands.items():
            con = compute_con(subepo, method, fmin=fmin, fmax=fmax)

            # flatten upper triangle
            for i in range(con.shape[0]):
                for j in range(i+1, con.shape[1]):
                    row = {
                        "con": con[j, i, 0],
                        "node1": chs[i],
                        "node2": chs[j],
                        "band": band
                    }
                    # add the group columns back in
                    if isinstance(combo, tuple):
                        for col, val in zip(group_cols, combo):
                            row[col] = val
                    else:
                        row[group_cols[0]] = combo

                    rows.append(row)

    return pd.DataFrame(rows)


def plot_connectivity_heatmap(
    df,
    # epochs,
    # method,
    hue,
    *,
    avg_level="all",
    freq_bands=None,
    vmin=0,
    vmax=1,
    upper=False,
    cmap="viridis",
    annot=True,
    fmt=".2f",
    figsize=None,
    **sns_kwargs,
):
    """
    For each unique hue-combination and each frequency band, plot
    a node x node heatmap of mean connectivity.

    Parameters
    ----------
    epochs : mne.Epochs
    method : str
        passed to `connectivity_df`
    hue : str or list of str
        same as in `connectivity_df`
    avg_level : {"all","subject"}
        how to pool epochs
    freq_bands : dict, optional
        band→(fmin,fmax). defaults to Delta…Gamma.
    vmin, vmax : float, optional
        color scale limits
    cmap : str
        matplotlib colormap
    annot : bool
        whether to write numbers in each square
    fmt : str
        text format for annotation
    figsize : tuple, optional
        e.g. (cols*4, rows*4)
    **sns_kwargs : passed to `sns.heatmap`
    """


    # 1) compute the tidy DataFrame
    # df = connectivity_df(epochs, method, hue,
    #                     avg_level=avg_level,
    #                     freq_bands=freq_bands)

    # 2) find unique groups & bands
    hue_cols = [hue] if isinstance(hue, str) else list(hue)
    groups   = df[hue_cols].drop_duplicates(keep="first")
    group_tuples = [
        tuple(row) if len(hue_cols)>1 else row[0]
        for row in groups.values
    ]
    band_order = list(freq_bands or {
        "Delta":(2,4),"Theta":(4,8),"Alpha":(8,13),
        "Beta":(13,30),"Gamma":(30,100)
    })

    # 3) node list
    nodes = sorted(set(df["node1"]) | set(df["node2"]))
    n = len(nodes)

    # 4) prep figure
    nrows = len(group_tuples)
    ncols = len(band_order)
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=figsize,
        squeeze=False
    )

    # 5) loop & plot
    for i, grp in enumerate(group_tuples):
        # subset df for this group
        if len(hue_cols)==1:
            sub = df[df[hue_cols[0]] == grp]
            title_prefix = f"{hue_cols[0]}={grp}"
        else:
            mask = np.all([
                df[col] == val
                for col, val in zip(hue_cols, grp)
            ], axis=0)
            sub = df[mask]
            title_prefix = ", ".join(f"{c}={v}" for c, v in zip(hue_cols, grp))

        for j, band in enumerate(band_order):
            ax = axes[i][j]
            subb = sub[sub["band"] == band]

            # build empty matrix
            mat = np.full((n, n), np.nan)
            for _, row in subb.iterrows():
                i1 = nodes.index(row["node1"])
                i2 = nodes.index(row["node2"])
                mat[i1, i2] = mat[i2, i1] = row["con"]
            
            mask_arr = None
            if upper:
                mask_arr = np.tril(np.ones((n,n), bool), k=-1)
            else:
                mask_arr = np.triu(np.ones((n,n), bool), k=1)

            sns.heatmap(
                mat,
                mask=mask_arr,
                ax=ax,
                vmin=vmin, vmax=vmax,
                cmap=cmap,
                annot=annot,
                fmt=fmt,
                xticklabels=nodes,
                yticklabels=nodes,
                cbar_kws={'shrink': 0.75},
                annot_kws={'size': 6},
                **sns_kwargs
            )
            ax.set_title(f"{title_prefix}\n{band}")
            ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    return fig, axes



def _p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def compute_connectivity_stats(
    df: pd.DataFrame,
    hue_plot: str,
    stats: str,
    band_order: list[str]
) -> pd.DataFrame:
    """
    Given a tidy connectivity DataFrame with columns
    [node1, node2, band, con, <hue_plot>],
    run the requested tests and return a DataFrame of results.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ["node1","node2","band","con", hue_plot].
    hue_plot : str
        Column name to treat as the grouping variable.
    stats : {"auto","ttest","anova","kruskal"}
        Which test(s) to run.
    band_order : list of band names
        The frequency bands to iterate over, in order.

    Returns
    -------
    stats_df : DataFrame with columns
        ["node1","node2","band","test","stat","p_value"]
    """
    import scipy.stats as ss
    records = []
    levels = sorted(df[hue_plot].unique())

    def do_test(arrs, test_name):
        if test_name == "ttest":
            return ss.ttest_ind(arrs[0], arrs[1],
                                equal_var=False, nan_policy="omit")
        elif test_name == "anova":
            return ss.f_oneway(*arrs)
        elif test_name == "kruskal":
            return ss.kruskal(*arrs)
        else:
            raise ValueError(f"Unknown stats='{stats}'")

    # decide test_name once for "auto"
    if stats == "auto":
        test_name = "ttest" if len(levels) == 2 else "anova"
    else:
        test_name = stats

    # loop node‐pairs × bands
    for (n1, n2), subpair in df.groupby(["node1","node2"]):
        for band in band_order:
            subb = subpair[subpair["band"] == band]
            arrs = [subb.loc[subb[hue_plot] == lvl, "con"].values
                    for lvl in levels]
            stat, p = do_test(arrs, test_name)
            records.append({
                "node1": n1,
                "node2": n2,
                "band":  band,
                "test":  test_name,
                "stat":  stat,
                "p_value": p
            })

    return pd.DataFrame(records)

def plot_connectivity_categorical(
    df,
    # epochs,
    # method,
    hue,
    *,
    avg_level="all",
    freq_bands=None,
    plot_type="box",       # "bar","box","violin"
    plot_individual_points=False,
    stats='auto',
    palette="hls",
    figsize=None,
    ylims = None,
    **sns_kwargs,
):
    """
    … same docstring …
    Supports multi-column hue by automatically combining them into a
    single synthetic column.
    """
    import warnings
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1) df
    # df = connectivity_df(epochs, method, hue,
    #                     avg_level=avg_level,
    #                     freq_bands=freq_bands)
    if freq_bands is None:
        freq_bands = {
            "Delta":(2,4),"Theta":(4,8),"Alpha":(8,13),
            "Beta":(13,30),"Gamma":(30,100)
        }

    # 2) create composite hue column
    hue_cols = [hue] if isinstance(hue, str) else list(hue)
    if len(hue_cols) > 1:
        df["_hue"] = df[hue_cols].agg(tuple, axis=1)
        hue_plot = "_hue"
        legend_title = ", ".join(hue_cols)
    else:
        hue_plot = hue_cols[0]
        legend_title = hue_plot

    # 3) pairs & bands
    pairs = df[["node1","node2"]].drop_duplicates()
    labels = [f"{r.node1}–{r.node2}" for r in pairs.itertuples(index=False)]
    band_order = list(freq_bands)

    # 4) warning
    if avg_level=="all" and plot_individual_points:
        warnings.warn("individual points only valid when avg_level='subject'")

    # if stats requested, compute once up front
    if stats is not None:
        stats_df = compute_connectivity_stats(df, hue_plot, stats, band_order)


    # 5) figure
    n = len(labels)
    nrows, ncols = row_col_layout(n)
    if figsize is None:
        figsize = (4.5*ncols, 3.5*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    # 6) loop & plot
    for ax, (node1, node2), lab in zip(axes, pairs.itertuples(index=False), labels):
        sub = df[(df.node1==node1)&(df.node2==node2)]
        if plot_type=="box":
            sns.boxplot(
                data=sub, x="band", y="con", hue=hue_plot,
                order=band_order, palette=palette,
                ax=ax, **sns_kwargs
            )
        elif plot_type=="bar":
            sns.barplot(
                data=sub, x="band", y="con", hue=hue_plot,
                order=band_order, palette=palette,
                errorbar="se", ax=ax, **sns_kwargs
            )
        else:  # violin
            sns.violinplot(
                data=sub, x="band", y="con", hue=hue_plot,
                order=band_order, palette=palette,
                cut=0, inner="box", ax=ax, **sns_kwargs
            )

        # individual points
        if plot_individual_points and avg_level=="subject":
            sns.stripplot(
                data=sub, x="band", y="con", hue=hue_plot,
                order=band_order, dodge=True,
                palette='dark:black', size=3, alpha=.5,
                jitter=False, ax=ax, legend=False
            )
            
        ax.set_ylim(ylims)

        # stats asterisks
        if stats is not None:
            # fix y‐coordinates at the top of the axes
            y_min, y_max = ax.get_ylim()
            y_line = y_max * 0.90     # 95% up the y‐axis
            y_text = y_max * 0.95     # 99% up for the stars

            substats = stats_df[
                (stats_df.node1 == node1) & (stats_df.node2 == node2)
            ]
            for b_i, band in enumerate(band_order):
                row   = substats[substats.band == band].iloc[0]
                stars = _p_to_stars(row.p_value)

                # horizontal line at constant height
                ax.plot([b_i - 0.2, b_i + 0.2],
                        [y_line, y_line],
                        lw=1.5, color="black")

                # text just above it
                ax.text(b_i, y_text, stars,
                        ha="center", va="bottom",
                        color="black")

        ax.set_title(lab)
        ax.set_xlabel("")
        ax.set_ylabel("Connectivity")
        sns.despine(ax=ax)

    # legend
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:len(df[hue_plot].unique())],
        labels_[:len(df[hue_plot].unique())],
        title=legend_title,
        bbox_to_anchor=(1.02,1), loc="upper left"
    )
    for ax in axes:
        if ax.get_legend(): ax.get_legend().remove()

    plt.tight_layout()
    
    if stats is not None:
        return fig, axes, stats_df
    return fig, axes

# Multivariate connectivity methods

def compute_multi_connectivity_df(
    epochs,
    method: str,
    hue,
    *,
    avg_level: str = "all",
    freq_bands: dict | None = None,
    fmin: float = 1.0,
    fmax: float = 100.0,
    mode: str = "multitaper",
    con_kwargs: dict | None = None,
):
    """
    Compute *multivariate* connectivity (e.g. MIM, MIC, CaCoh…) for each
    group or subject and return a tidy DataFrame.

    Parameters
    ----------
    epochs : mne.Epochs
        The data.
    method : str
        Multivariate connectivity estimator (e.g. "mim", "mic", "cacoh", …).
    hue : str or list[str]
        Metadata column(s) defining experimental groups.
    avg_level : {"all","subject"}, default "all"
        - "all":    pool all epochs matching each unique hue combination
                    and compute one connectivity spectrum per combo.
        - "subject": compute one connectivity spectrum per subject
                    (metadata column "animal_id" must exist), retaining
                    hue labels as well.
    freq_bands : dict | None
        Mapping band_name → (fmin, fmax). Defaults to
        {"Delta":(2,4),…"Gamma":(30,100)}.
        Only used by downstream plotting; this function always returns
        freq-by-freq rows.
    fmin, fmax : float, default (1,100)
        Frequency bounds for `spectral_connectivity_epochs`.
    mode : str, default "multitaper"
        Backend mode for spectral estimation.
    con_kwargs : dict, optional
        Extra keyword-args forwarded to `spectral_connectivity_epochs`.

    Returns
    -------
    pd.DataFrame
        Columns:
          - every column in `hue` (and, if avg_level="subject", also "animal_id"),
          - "freq" (float),
          - "con" (connectivity value).
    """
    if freq_bands is None:
        freq_bands = {
            "Delta": (2, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta":  (13, 30),
            "Gamma": (30, 100),
        }

    con_kwargs = con_kwargs or {}

    # validate hue columns
    hue_cols = [hue] if isinstance(hue, str) else list(hue)
    missing = [c for c in hue_cols if c not in epochs.metadata.columns]
    if missing:
        raise ValueError(f"Hue column(s) not in metadata: {missing}")

    # decide how to group
    if avg_level == "subject":
        if "animal_id" not in epochs.metadata.columns:
            raise ValueError("avg_level='subject' requires 'animal_id' column")
        group_cols = ["animal_id"] + hue_cols
    elif avg_level == "all":
        group_cols = hue_cols
    else:
        raise ValueError("avg_level must be 'all' or 'subject'")

    records = []
    ch_names = epochs.ch_names

    # loop over each unique grouping
    for combo, submeta in tqdm(epochs.metadata.groupby(group_cols), desc="Computing multivariate connectivity"):
        # pick the matching epochs
        mask = np.ones(len(epochs), bool)
        for col, val in zip(group_cols, (combo if isinstance(combo, tuple) else [combo])):
            mask &= (epochs.metadata[col] == val)
        sub_epochs = epochs[mask]

        # compute connectivity (multivariate) → con_mat shape = (1, n_freqs)
        con_obj = spectral_connectivity_epochs(
            sub_epochs,
            method=method,
            mode=mode,
            sfreq=sub_epochs.info["sfreq"],
            fmin=fmin,
            fmax=fmax,
            faverage=False,
            verbose="ERROR",
            **con_kwargs,
        )
        data = con_obj.get_data()[0, :]  # shape: (n_freqs,)

        freqs = con_obj.freqs  # array of shape (n_freqs,)

        # extract group‐labels as dict
        grp_dict = {}
        if isinstance(combo, tuple):
            for col, val in zip(group_cols, combo):
                grp_dict[col] = val
        else:
            grp_dict[group_cols[0]] = combo

        # append one row per frequency
        for f, cval in zip(freqs, data):
            row = {"freq": float(f), "con": float(cval), **grp_dict}
            records.append(row)

    return pd.DataFrame(records)

def plot_multi_connectivity_spectrum(
    df: pd.DataFrame,
    hue,
    *,
    avg_level="all",
    err_method="sem",             # {"sd","sem","ci",None}
    plot_individual_points=False,
    palette="hls",
    figsize=None,
):
    """
    Line plot of multivariate connectivity vs. frequency, with optional error bands.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `compute_multi_connectivity_df`. Must contain columns:
        ["freq","con", <hue>] and, if avg_level="subject", "animal_id".
    hue : str or list[str]
        Metadata column(s) used as the grouping variable.
    avg_level : {"all","subject"}, default "all"
        If "subject", df has one row per subject/frequency, and we compute mean ± error across subjects.
        If "all", df has one row per group/frequency and no error is drawn.
    err_method : {"sd","sem","ci", None}, default "sem"
        If not None and avg_level="subject", compute and plot error bands:
          - "sd":   standard deviation across subjects
          - "sem":  standard error of the mean
          - "ci":   95% confidence interval (1.96*SEM)
    plot_individual_points : bool
        When avg_level="subject", also plot each subject’s spectrum faintly.
    palette : str or sequence
        Seaborn palette for the group means.
    log_freq : bool, default False
        If True, x-axis (frequency) is log-scaled.
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    # 0) Composite hue if needed
    hue_cols = [hue] if isinstance(hue, str) else list(hue)
    if len(hue_cols) > 1:
        df["_hue"] = df[hue_cols].agg(tuple, axis=1)
        hue_plot = "_hue"
        legend_title = ", ".join(hue_cols)
    else:
        hue_plot = hue_cols[0]
        legend_title = hue_plot

    # 1) Determine unique groups
    groups = sorted(df[hue_plot].unique())
    colours = sns.color_palette(palette, n_colors=len(groups))
    pal = {g: c for g, c in zip(groups, colours)}

    # 2) Prepare figure
    if figsize is None:
        figsize = (8, 5)
    fig, ax = plt.subplots(figsize=figsize)

    # 3) For each group, compute freq‐wise mean and error
    for grp in groups:
        subg = df[df[hue_plot] == grp]

        if avg_level == "subject":
            # Pivot: rows=subjects, cols=freq, values=con
            pivot = (
                subg
                .pivot_table(
                    index="animal_id",
                    columns="freq",
                    values="con",
                    aggfunc="mean"
                )
            )
            # Sort frequencies
            freqs = np.array(pivot.columns)
            # Each row = one subject's spectrum
            subj_arr = pivot.values            # shape = (n_subj, n_freqs)

            # Optional: plot each subject's trace faintly
            if plot_individual_points:
                for row in subj_arr:
                    ax.plot(
                        freqs,
                        row,
                        color=pal[grp],
                        alpha=0.25,
                        linewidth=0.8
                    )

            # Compute group‐mean and error across subjects
            mean_vals = np.nanmean(subj_arr, axis=0)
            if err_method == "sd":
                err_vals = np.nanstd(subj_arr, axis=0)
            elif err_method == "sem":
                nsub = np.sum(~np.isnan(subj_arr), axis=0)
                err_vals = np.nanstd(subj_arr, axis=0) / np.sqrt(nsub)
            elif err_method == "ci":
                nsub = np.sum(~np.isnan(subj_arr), axis=0)
                sem = np.nanstd(subj_arr, axis=0) / np.sqrt(nsub)
                err_vals = 1.96 * sem
            else:
                err_vals = None

        else:  # avg_level == "all"
            # Each group has exactly one "con" per freq
            grouped = subg.groupby("freq")["con"].mean()
            freqs = np.array(grouped.index)
            mean_vals = grouped.to_numpy()
            err_vals = None

        # 4) Plot the mean line (cast label to string to avoid tuple warning)
        ax.plot(
            freqs,
            mean_vals,
            color=pal[grp],
            linewidth=2,
            label=str(grp),
        )

        # 5) Fill error band if requested
        if err_vals is not None:
            ax.fill_between(
                freqs,
                mean_vals - err_vals,
                mean_vals + err_vals,
                alpha=0.25,
                color=pal[grp],
            )

    # 6) Final styling
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Connectivity")
    ax.legend(title=legend_title, loc="best")
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig, ax


def plot_multi_connectivity_band(
    df: pd.DataFrame,
    hue,
    *,
    avg_level="all",
    freq_bands: dict | None = None,
    plot_type="box",               # "box","bar","violin"
    plot_individual_points=False,
    stats="auto",                  # None|"auto"|"ttest"|"anova"|"kruskal"
    palette="hls",
    figsize=None,
    **sns_kwargs,
):
    """
    Categorical plot (bar/box/violin) of multivariate connectivity
    aggregated into frequency bands, with optional stats.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `compute_multi_connectivity_df`. Must contain:
        ["freq","con", <hue>] and, if avg_level="subject", "animal_id".
    hue : str or list[str]
        Metadata column(s) used as the grouping variable.
    avg_level : {"all","subject"}, default "all"
        How the dataframe was generated: if "subject", df has one row
        per subject/frequency; if "all", df has one row per group/frequency.
    freq_bands : dict | None
        Mapping band→(fmin,fmax). Defaults to Delta…Gamma.
    plot_type : {"box","bar","violin"}, default "box"
        Which categorical plot to draw.
    plot_individual_points : bool
        Overlay subject-level points (only when avg_level="subject").
    stats : None | "auto" | "ttest" | "anova" | "kruskal"
        If not None, run the indicated test *per band* across hue-levels.
        - "auto": 2 levels→ttest (Welch); >2→ANOVA.
    palette : seaborn palette name or list
    figsize : tuple, optional
    **sns_kwargs : passed to the seaborn call

    Returns
    -------
    fig, ax (, stats_df)
        If `stats` is not None, returns a third output `stats_df` with
        columns ["band","test","stat","p_value"].
    """
    # 0) Default freq_bands
    if freq_bands is None:
        freq_bands = {
            "Delta": (2, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta":  (13, 30),
            "Gamma": (30, 100),
        }

    # 1) Build a composite hue column if needed
    hue_cols = [hue] if isinstance(hue, str) else list(hue)
    if len(hue_cols) > 1:
        df["_hue"] = df[hue_cols].agg(tuple, axis=1)
        hue_plot = "_hue"
        legend_title = ", ".join(hue_cols)
    else:
        hue_plot = hue_cols[0]
        legend_title = hue_plot

    # 2) Assign each frequency to a band
    def freq_to_band(f):
        for band, (lo, hi) in freq_bands.items():
            if lo <= f < hi:
                return band
        return np.nan

    df["band"] = df["freq"].apply(freq_to_band)
    band_order = list(freq_bands.keys())

    # 3) Collapse to one row per (hue_plot, [animal_id], band)
    if avg_level == "subject":
        # One row per subject × hue × band
        df_band = (
            df
            .groupby([hue_plot, "animal_id", "band"], observed=True)["con"]
            .mean()
            .reset_index()
        )
    else:  # avg_level == "all"
        # One row per hue × band
        df_band = (
            df
            .groupby([hue_plot, "band"], observed=True)["con"]
            .mean()
            .reset_index()
        )

    # 4) Warn if individual points requested but avg_level="all"
    import warnings
    if avg_level == "all" and plot_individual_points:
        warnings.warn(
            "plot_individual_points only valid when avg_level='subject'."
        )

    # 5) Prepare figure
    if figsize is None:
        figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)

    # 6) Plot categorical
    if plot_type == "box":
        sns.boxplot(
            data=df_band,
            x="band", y="con", hue=hue_plot,
            order=band_order,
            palette=palette,
            ax=ax,
            **sns_kwargs
        )
    elif plot_type == "bar":
        sns.barplot(
            data=df_band,
            x="band", y="con", hue=hue_plot,
            order=band_order,
            palette=palette,
            errorbar="se",
            ax=ax,
            **sns_kwargs
        )
    else:  # "violin"
        sns.violinplot(
            data=df_band,
            x="band", y="con", hue=hue_plot,
            order=band_order,
            palette=palette,
            cut=0,
            inner="box",
            ax=ax,
            **sns_kwargs
        )

    # 7) Overlay individual‐subject points if requested
    if plot_individual_points and avg_level == "subject":
        sns.stripplot(
            data=df_band,
            x="band", y="con", hue=hue_plot,
            order=band_order,
            dodge=True,
            color="palette='dark:black",
            size=3,
            alpha=0.5,
            jitter=False,
            ax=ax,
            legend=False
        )

    # 8) Run stats if requested and annotate
    stats_df = None
    if stats is not None:
        # Only meaningful if avg_level == "subject"
        if avg_level != "subject":
            warnings.warn(
                "Cannot run stats when avg_level='all'; "
                "skipping statistical tests."
            )
        else:
            # Gather unique hue‐levels
            levels = sorted(df_band[hue_plot].unique())

            # Decide test once for "auto"
            if stats == "auto":
                test_name = "ttest" if len(levels) == 2 else "anova"
            else:
                test_name = stats

            records = []
            for band in band_order:
                subb = df_band[df_band["band"] == band]
                arrs = [
                    subb.loc[subb[hue_plot] == lvl, "con"].values
                    for lvl in levels
                ]
                # Choose test
                if test_name == "ttest":
                    stat, pval = stats.ttest_ind(
                        arrs[0], arrs[1], equal_var=False, nan_policy="omit"
                    )
                elif test_name == "anova":
                    stat, pval = stats.f_oneway(*arrs)
                elif test_name == "kruskal":
                    stat, pval = stats.kruskal(*arrs)
                else:
                    raise ValueError(f"Unknown stats='{stats}'")

                # record
                records.append({
                    "band": band,
                    "test": test_name,
                    "stat": stat,
                    "p_value": pval
                })

                # annotate stars at top
                stars = _p_to_stars(pval)
                y_min, y_max = ax.get_ylim()
                y_line = y_max * 0.95
                y_text = y_max * 0.99
                b_i = band_order.index(band)

                ax.plot(
                    [b_i - 0.2, b_i + 0.2],
                    [y_line, y_line],
                    lw=1.5, color="black"
                )
                ax.text(
                    b_i,
                    y_text,
                    stars,
                    ha="center",
                    va="bottom",
                    color="black"
                )

            stats_df = pd.DataFrame(records)

    # 9) Final formatting
    ax.set_xlabel("Band")
    ax.set_ylabel("Connectivity")
    sns.despine(ax=ax)

    # shared legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[: len(df_band[hue_plot].unique())],
        labels[: len(df_band[hue_plot].unique())],
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.get_legend().remove()

    plt.tight_layout()

    if stats_df is not None:
        return fig, ax, stats_df
    return fig, ax