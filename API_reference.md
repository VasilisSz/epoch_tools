# API Reference

---

## `compare_psd`

```python
Epochs.compare_psd(
    hue,
    *,
    channels: Union[List[str], str] = "all",
    method: str = "welch",
    avg_level: str = "subject",
    plot_individual_points: bool = False,
    bad_channels: Optional[Dict[Union[str, int], List[str]]] = None,
    err_method: Optional[str] = "sem",
    hue_order: Optional[List[Union[str, Tuple]]] = None,
    plot_type: str = "line",
    palette: str = "hls",
    **kwargs
) → Tuple[matplotlib.figure.Figure, numpy.ndarray]
```

Compare and visualize power spectral density (PSD) across experimental groups.

**Parameters:**

* **`hue`**

  * **Type:** `str` or `list[str]`
  * **Description:** Column name(s) in the `Epochs.metadata` DataFrame that define grouping.

    * If a single string, groups by that column.
    * If a list of strings, forms composite (tuple) groups.
  * **Required.**

* **`channels`**

  * **Type:** `list[str]` or `str`
  * **Default:** `"all"`
  * **Description:**

    * `"all"`: use all channels in the wrapped `mne.Epochs`.
    * Otherwise, a list of channel names to include in PSD computation (e.g., `["OFC_R", "S1_L"]`).

* **`method`**

  * **Type:** `str`
  * **Default:** `"welch"`
  * **Options:**

    * `"welch"`: use `mne.time_frequency.psd_array_welch`.
    * `"multitaper"`: use `mne.time_frequency.psd_array_multitaper`.
  * **Description:** PSD estimation algorithm. Additional options (e.g., `bandwidth`, `adaptive`) can be passed via `**kwargs`.

* **`avg_level`**

  * **Type:** `str`
  * **Default:** `"subject"`
  * **Options:**

    * `"subject"`: for each group, first average PSD across epochs *per subject* (requires `"animal_id"` in metadata), then average those subject‐means.
    * `"all"`: pool all epochs per group and compute PSD average across them.
  * **Description:** Level at which to average before plotting.

* **`plot_individual_points`**

  * **Type:** `bool`
  * **Default:** `False`
  * **Description:**

    * When `avg_level="subject"`, overlay each subject’s PSD (one faint line per subject).
    * When `avg_level="all"`, overlay each epoch’s PSD (warning issued).

* **`bad_channels`**

  * **Type:** `dict[Union[str,int], list[str]]` or `None`
  * **Default:** `None`
  * **Description:**

    * Maps `animal_id → [channel_names]` to exclude.
    * A key of `None` indicates channels excluded for *all* animals.
    * PSD values in any “bad” channel are set to `NaN` and omitted from averaging.

* **`err_method`**

  * **Type:** `str` or `None`
  * **Default:** `"sem"`
  * **Options:**

    * `"sd"` – standard deviation across subjects (or epochs when `avg_level="all"`).
    * `"sem"` – standard error of the mean (SD/√N).
    * `"ci"` – 95% confidence interval (1.96 × SEM).
    * `None` – no error shading.
  * **Description:** How to compute and display error bands around each group‐mean PSD.

* **`hue_order`**

  * **Type:** `list` or `None`
  * **Default:** `None`
  * **Description:** Explicit ordering of group labels.

    * For single‐column `hue`, provide a list of values (e.g., `["WT","KO"]`).
    * For composite `hue` (multiple columns), provide a list of tuples in desired order.
    * If `None`, groups are sorted alphanumerically.

* **`plot_type`**

  * **Type:** `str`
  * **Default:** `"line"`
  * **Options:**

    * `"line"`: one subplot per channel showing PSD vs frequency; one line per group, optional error‐band/individual lines.
    * `"box"`, `"bar"`, `"violin"`: collapse each channel’s PSD into canonical bands (`delta`,`theta`,`alpha`,`beta`,`gamma`), then produce one subplot per channel with band‐wise distribution plots.
  * **Description:** Choose plot style.

* **`palette`**

  * **Type:** `str` or color palette object
  * **Default:** `"hls"`
  * **Description:** Seaborn or Matplotlib palette to color group levels.

* **`**kwargs`**

  * **Type:** Additional keyword arguments
  * **Description:** Passed directly to underlying PSD estimator (`psd_array_welch` or `psd_array_multitaper`). For example, when `method="multitaper"`, you might specify `bandwidth=4` or `adaptive=True`.

**Returns:**

* `(fig, axes)`

  * `fig`: `matplotlib.figure.Figure`.
  * `axes`: 1D array of `matplotlib.axes.Axes`, one per channel.

---

## `compare_con`

```python
Epochs.compare_con(
    hue,
    *,
    method: str,
    plot_type: str,
    avg_level: str = "all",
    freq_bands: Optional[Dict[str, Tuple[float,float]]] = None,
    bad_channels: Optional[Dict[Union[str,int], List[str]]] = None,
    multivariate_nodes: Optional[List[str]] = None,
    plot_individual_points: bool = False,
    stats: Optional[str] = None,
    err_method: Optional[str] = "sem",
    palette: str = "hls",
    figsize: Optional[Tuple[float, float]] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    upper: bool = False,
    ylims: Optional[Tuple[float, float]] = None,
    con_kwargs: Optional[Dict] = None
) → Union[
    Tuple[matplotlib.figure.Figure, numpy.ndarray],
    Tuple[matplotlib.figure.Figure, numpy.ndarray, pandas.DataFrame],
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes],
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, pandas.DataFrame]
]
```

Compare and visualize functional connectivity (bivariate or multivariate) across experimental groups.

### Parameters

* **`hue`**

  * **Type:** `str` or `list[str]`
  * **Description:** Column name(s) in `Epochs.metadata` defining grouping.

    * Single string: group by that column.
    * List of strings: form composite tuple‐groups.
  * **Required.**

* **`method`**

  * **Type:** `str`
  * **Description:** Connectivity measure to compute.

    * **Bivariate methods:**

      ```
      "coh", "cohy", "imcoh", "plv", "pli", "dpli",
      "wpli", "wpli2_debiased", "ppc", …
      ```
    * **Multivariate methods:**

      ```
      "mim", "mic", "cacoh", "gc", "gc_tr"
      ```
    * If `method` is in the multivariate set, the call uses multivariate workflows; otherwise, bivariate workflows.

* **`plot_type`**

  * **Type:** `str`
  * **Description:** Determines which plotting routine is used.

    * **Bivariate only** (`method` not in multivariate set):

      * `"heatmap"`: node×node connectivity matrices for each group × band.
      * `"box"`, `"bar"`, `"violin"`: categorical per node‐pair × band.
    * **Multivariate only** (`method` in `{ "mim","mic","cacoh","gc","gc_tr" }`):

      * `"spectrum"`: plot connectivity vs frequency (one curve per group, ±error).
      * `"box"`, `"bar"`, `"violin"`: categorical per band.

* **`avg_level`**

  * **Type:** `str`
  * **Default:** `"all"`
  * **Options:**

    * `"all"`: pool all epochs per group and compute a single connectivity spectrum or single band‐value per group.
    * `"subject"`: compute connectivity separately for each subject (`"animal_id"` required).

      * Bivariate: treat each subject’s band‐averaged connectivity as one sample.
      * Multivariate spectrum: plot subject‐wise spectra then average across subjects.
  * **Description:** Level at which to perform grouping/averaging.

* **`freq_bands`**

  * **Type:** `dict[str, (float, float)]` or `None`
  * **Default:**

    ```python
    {
      "Delta": (2, 4),
      "Theta": (4, 8),
      "Alpha": (8, 13),
      "Beta":  (13, 30),
      "Gamma": (30, 100),
    }
    ```
  * **Description:** Frequency‐band boundaries for categorical plots (`"box"`, `"bar"`, `"violin"`). Ignored for `"heatmap"` and `"spectrum"`.

* **`bad_channels`**

  * **Type:** `dict[Union[str,int], list[str]]` or `None`
  * **Default:** `None`
  * **Description:** Channels to exclude from connectivity calculation.

    * Key `None`: global bad channels (drop for all subjects).
    * Key `animal_id`: drop only those channels for that subject.
  * **Behavior:**

    * **Bivariate:** drop any row where `node1` or `node2` is a “bad” channel (global or subject‐specific).
    * **Multivariate:**

      1. Remove all globally bad channels from the selected node set.
      2. Exclude any subject whose personal bad‐channel list intersects the final node set (no partial connectivity computation).

* **`multivariate_nodes`**

  * **Type:** `list[str]` or `None`
  * **Default:** `None`
  * **Description:**

    * Only for multivariate methods.
    * If provided, restricts connectivity to these channel names (minus any globally bad channels).
    * Subjects whose personal bad channel list overlaps this set are excluded entirely.
    * If `None`, use all available (non‐excluded) channels.

* **`plot_individual_points`**

  * **Type:** `bool`
  * **Default:** `False`
  * **Description:**

    * **Bivariate categorical:** overlay each subject’s band‐averaged value (only valid if `avg_level="subject"`; warning if `"all"`).
    * **Multivariate spectrum:** overlay each subject’s full‐spectrum trace faintly.
    * **Multivariate categorical:** overlay each subject’s band‐averaged connectivity.

* **`stats`**

  * **Type:** `str` or `None`
  * **Default:** `None`
  * **Options:**

    * `None`: skip statistical testing.
    * `"auto"`: two levels → Welch’s t‐test; >2 → one‐way ANOVA.
    * `"ttest"`: Welch’s t‐test (two groups).
    * `"anova"`: one‐way ANOVA (>2 groups).
    * `"kruskal"`: Kruskal–Wallis nonparametric test (>2 groups).
  * **Description:** If not `None`, run per‐band tests across group levels and annotate plots with significance asterisks (`"*","**","***"`). Returns a DataFrame of test results if requested.

* **`err_method`**

  * **Type:** `str` or `None`
  * **Default:** `"sem"`
  * **Options:**

    * `"sd"` – standard deviation across subjects (only for multivariate `"spectrum"`).
    * `"sem"` – standard error of the mean.
    * `"ci"` – 95% confidence interval (1.96 × SEM).
    * `None` – no error shading.
  * **Description:** Used only for `plot_type="spectrum"` in multivariate mode.

* **`palette`**

  * **Type:** `str` or palette object
  * **Default:** `"hls"`
  * **Description:** Seaborn/Matplotlib palette for coloring group levels.

* **`figsize`**

  * **Type:** `tuple[float, float]` or `None`
  * **Default:** `None`
  * **Description:** Figure size in inches. If `None`, default layout is chosen according to number of subplots.

* **`vmin`, `vmax`**

  * **Type:** `float`
  * **Default:** `0.0`, `1.0`
  * **Description:** Lower/upper bounds of color scale; only used for `plot_type="heatmap"`.

* **`upper`**

  * **Type:** `bool`
  * **Default:** `False`
  * **Description:**

    * Only for `plot_type="heatmap"`.
    * If `True`, mask the lower triangle (show upper half).
    * If `False`, mask the upper triangle (show lower half).

* **`ylims`**

  * **Type:** `Tuple[float, float]` or `None`
  * **Default:** `None`
  * **Description:**

    * Only for categorical plots (`"box"`, `"bar"`, `"violin"`).
    * Sets the same y‐axis limits for all node‐pair subplots.

* **`con_kwargs`**

  * **Type:** `dict` or `None`
  * **Default:** `None`
  * **Description:** Additional keyword arguments passed directly to `mne_connectivity.spectral_connectivity_epochs(...)`, e.g. `{"n_jobs": 4}`.

**Returns:**

* **Bivariate + `plot_type="heatmap"`** → `(fig, axes)`

  * `axes`: 2D array of subplots shaped `(n_groups, n_bands)`
* **Bivariate + categorical (`"box"`, `"bar"`, `"violin"`)** →

  * `(fig, axes)` if `stats is None`, or
  * `(fig, axes, stats_df)` if `stats` provided
  * `axes`: 1D array of subplots, one per node‐pair
* **Multivariate + `plot_type="spectrum"`** → `(fig, ax)`

  * Single Axes: connectivity vs frequency
* **Multivariate + categorical (`"box"`, `"bar"`, `"violin"`)** →

  * `(fig, ax)` if `stats is None`, or
  * `(fig, ax, stats_df)` if `stats` provided

---

### Notes

* **Caching:**

  * The connectivity DataFrame is cached within `self.con_results` using a key of `(method, hue_cols, avg_level, freq_bands, bad_channels, multivariate_nodes, con_kwargs)`.
  * Subsequent calls with identical arguments reuse the cached results to avoid recomputation.

* **Bad channels – Bivariate:**

  * Dropping a channel for a particular `animal_id` removes any row in the tidy connectivity DataFrame where that subject’s `node1` or `node2` equals the “bad” channel.
  * A key of `None` in `bad_channels` indicates channels dropped for all subjects (global).

* **Bad channels – Multivariate:**

  1. **Global bads** (`bad_channels[None]`) are removed from the chosen node set.
  2. Any subject whose personal bad list intersects the final node set is excluded entirely (cannot compute multivariate connectivity on incomplete node set).

* **Frequency bands (Bivariate categorical & Multivariate categorical):**

  * Defined by the `freq_bands` dictionary.
  * Bivariate categorical: for each node‐pair and each subject (or pooled group), compute average connectivity within each band.
  * Multivariate categorical: for each subject or pooled group, compute average connectivity across all channels in `multivariate_nodes` for that band.

* **Statistical testing (WIP: DOES NOT WORKING PROPERLY RIGHT NOW):**

  * Per‐band comparisons across hue levels.
  * Bivariate: tests run separately for each node‐pair × band combination.
  * Multivariate categorical: tests run per band across hue levels.
  * Significance annotated as `*` (< .05), `**` (< .01), `***` (< .001), or `ns`.
  * If `avg_level="all"` and `stats` requested, a warning is issued (no subject‐wise replication).

---
