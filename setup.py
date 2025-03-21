from setuptools import setup, find_packages
print(find_packages())
setup(
    name="epoch_tools",
    version="0.1.1",
    author="Vasilis Siozos",
    author_email="v.siozos@rug.nl",
    description="A package for analyzing and clustering EEG epochs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VasilisSz/epoch_tools",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "mne",
        "numpy",
        "pandas",
        "matplotlib",
        "PyQt5",
        "seaborn",
        "scikit-learn",
        "shap",
        "umap-learn",
        "hdbscan",
        "pca",
        "ipywidgets",
        "IPython",
        "shap",
        "xgboost",
        "catboost",
        "bayesian-optimization"

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
