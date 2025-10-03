# `epoch_tools` 🧠⚡
`epoch_tools` is a Python package extending MNE's `Epochs` for advanced EEG analysis, focusing on feature clustering, visualization, and interpretability. Currently in active development, with new features and improvements planned. Contributions and feedback are welcome!
_____

## Key Features
- **Feature Clustering**: Apply KMeans, HDBSCAN, or custom clustering after optional dimensionality reduction (UMAP, PCA, t-SNE).
- **Interpretability Tools**: SHAP integration, feature importance analysis, and cluster visualization.
- **MNE Integration**: Full compatibility with MNE's Epochs API. Seamlessly save/load data with metadata and clusters.
- **Visualization**: Built-in plotting for PSDs, feature distributions, cluster projections, for easy exploratory analysis of `Epochs`.

## Installation
```bash
pip install git+https://github.com/VasilisSz/epoch_tools.git
```

To update to the latest version

```bash
pip install --upgrade git+https://github.com/VasilisSz/epoch_tools.git
```

## Tutorials
For detailed examples, see the tutorials directory, containing jupyter notebooks, showing the full functionality of the package, including

- Interactive feature selection
- Dimensionality reduction and unsupervised clustering
- Cluster interpretation

## To-do

Some planned features that will be included soon:
- Raw .edf file preprocessing into `Epochs`
- Feature engineering on EEG Epochs
- Expanded Epoch analysis options
- Integration of deep clustering methods (e.g. VaDE)
- Pipeline 