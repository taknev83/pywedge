# Welcome to Pywedge documentation

## What is Pywedge?
Pywedge is a [pip installable](https://pypi.org/project/pywedge/) Python package that intends to,

1. Make multiple charts in a single line of code, to enable the user to quickly read through the charts and can make informed choices in pre-processing steps

2. Quickly preprocess the data by taking the user’s preferred choice of pre-processing techniques & it returns the cleaned datasets to the user in the first step.

3. Make a baseline model summary, which can return ten various baseline models, which can point the user to explore the best performing baseline model.

Pywedge intends to help the user by quickly making charts, preprocessing the data and to rightly point out the best performing baseline model for the given dataset so that the user can spend quality time tuning such a model algorithm.

## Installation


```
pip install pywedge
```

For JupyterLab, please run the following commands in anaconda prompt to enable required JupyterLab extensions to display interactive chart widget,

```
conda install -c conda-forge nodejs

jupyter labextension install @jupyter-widgets/jupyterlab-manager

jupyter labextension install jupyterlab-plotly@4.14.1

jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1
```

Mlflow is not a hard requirement in Pywedge, this is mainly to keep the pywedge light weight for the benefit of trying pywedge in web apps like Heroku. So mlflow has to be separately installed to track the hyperparameters,

```
pip install mlflow
```

Please note: Pywedge is available only in Jupyter Notebook & JupyterLab environments

## Pywedge Features
Cleans the raw data frame to fed into ML models. Following data pre_processing will be carried out,

* Makes 8 different types of ***interactive charts*** with interactive axis selection widgets

* Interactive pre-processing & 10 different baseline models
    * Missing values imputation for numeric & categorical columns
    * Standardization
    * Feature importance
    * Class oversampling using SMOTE
    * Computes 10 different baseline models

 * Interactive Hyperparameter tuning & tracking hyperparameters using integreted MLFlow
    * Classification / Regression Hyperparameters tuning
        * Available baseline estimators for interactive hyperparameter tuning as of now, more baseline estimators will be added soon for interactive hyperparameter tunings
        * Logistic / Linear Regression
        * Decision Tree Classifier / Regressor
        * Random Forest Classifier/ Regressor
        * KNN Classifier / Regressor
