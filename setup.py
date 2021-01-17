

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywedge",
    version="0.5.1.7",
    author="Venkatesh Rengarajan Muthu",
    author_email="taknev83@gmail.com",
    description="Makes Interactive Charts, Interactive Baseline, Interactive Hyperparameter Tuning & Predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/taknev83/pywedge",
    packages=['pywedge'],
    include_package_data=True,
    install_requires=[
        "xgboost",
        "pandas",
        "catboost>0.24",
        "numpy",
        "scikit-learn",
        "imbalanced-learn",
        "plotly",
        "ipywidgets",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
