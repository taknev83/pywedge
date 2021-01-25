[![Downloads](https://pepy.tech/badge/pywedge)](https://pepy.tech/project/pywedge) [![PyPI version](https://badge.fury.io/py/pywedge.svg)](https://badge.fury.io/py/pywedge) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

# Pywedge        

# [Docs](https://taknev83.github.io/pywedge-docs/) | [PyPi](https://pypi.org/project/pywedge/)

[![Try Pywedge In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gqe6RzIe99NTsUbD3oLPrBVI5dqo5cUz?usp=sharing)

## [Pywedge_Demo_Heroko Web App](https://pywedge-demo.herokuapp.com/)

# [>> Pywedge Documentation](https://taknev83.github.io/pywedge-docs/)

## What is Pywedge?

Pywedge is a [pip installable](https://pypi.org/project/pywedge/) Python package that intends to,

1. Make multiple interactive charts in a single line of code, to enable the user to quickly read through the charts and can make informed choices in pre-processing steps

2. Interactively preprocess the data by taking the user’s preferred choice of pre-processing techniques,

3. Make a baseline model summary, which can return ten various baseline models & predict the standout test data from selected baseline model.

4. Interactively select hyperparameters in a widget style tab, track the hyperparameters using MLFlow & predict on standout data.

Pywedge intends to help the user by quickly making charts, preprocessing the data and to rightly point out the best performing baseline model for the given dataset so that the user can spend quality time tuning such a model algorithm.

# Installation

```
pip install pywedge --upgrade
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

# Pywedge Features
Cleans the raw data frame to fed into ML models. Following data pre_processing will be carried out,
1) Makes 8 different types of ***interactive charts*** with interactive axis selection widgets
2) Interactive pre-processing & 10 different baseline models 
    - Missing values imputation for numeric & categorical columns
    - Standardization
    - Feature importance
    - Class oversampling using SMOTE
    - Computes 10 different baseline models
 3) Interactive Hyperparameter tuning & tracking hyperparameters using integreted MLFlow
    - Classification / Regression Hyperparameters tuning
        - Available baseline estimators for interactive hyperparameter tuning as of now, more baseline estimators will be added soon for interactive hyperparameter tunings
        
        | Classification | Regression |
        | :---: | :---: |
        | Logistic Regression | Linear Regression |
        | Decision Tree Classifier | Decision Tree Regressor | 
        | Random Forest Classifier | Random Forest Regressor |
        | AdaBoost Classifier | AdaBoost Regressor |
        | ExtraTrees Classifier | ExtraTrees Regressor |
        | KNN Classifier | KNN Regressor |
              
# Make_Charts()
Makes 8 different types of interactive Charts with interactive axis selection widgets in a single line of code for the given dataset. 

Different types of Charts viz,
1) Scatter Plot
2) Pie Chart
3) Bar Plot
4) Violin Plot
5) Box Plot
6) Distribution Plot
7) Histogram 
8) Correlation Plot
    
Arguments:
1) Dataframe
2) c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
3) y = target column name as a string 
        
Returns:

Charts widget

Pywedge-Make_Charts Demo YouTube link below,

<div align="left">
      <a href="https://youtu.be/-3rrQqyMTVk">
     <img 
      src="https://raw.githubusercontent.com/taknev83/pywedge/main/images/mq1.jpg" 
      alt="Pywedge-Make_Charts" 
      style="width:100%;">
      </a>
    </div>


Please read about Pywedge-Make_Charts module in this article published in [Analytics India Magazine](https://analyticsindiamag.com/how-to-build-interactive-eda-in-2-lines-of-code-using-pywedge/).

# baseline_model()
The baseline_model class starts with interactive pre-processing steps,
![baseline_model](https://raw.githubusercontent.com/taknev83/pywedge/main/images/baseline_models_inputs.jpg)

Instantiate the baseline class & call the classification_summary method from baseline_model class,

```python
blm = pw.baseline_model(train, test, c, y, type)
blm.classification_summary()
```

Args:
1) train = train dataframe
2) test = test dataframe
3) c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
4) y = target column name as a string 
5) type = Classification(Default) / Regression


- For classification - classification_summary() 
- For Regression - Regression_summary()

User Inputs:
1) Categorical columns conversion options
    -   Using Pandas Catcodes
    -   Using Pandas Get Dummies
2) Standardization Options,
    -   Standard scalar
    -   Minmax scalar
    -   Robust Scalar
    -   No Standardization
3) For Classification, Class balance using SMOTE options
    -   Yes
    -   No
4) Test Size for Train-test split
    -   test size in float

Returns:

1) Baseline models tab - Various baseline model metrics
2) Predict Baseline model tab - User can select the preferred available baseline choices to predict

![baseline_output](https://raw.githubusercontent.com/taknev83/pywedge/main/images/baseline_model_output.gif)


# Pywedge_HP()

* Introducing interactive hyperparameter tuning classes, Pywedge_HP, which has following two methods,
    - HP_Tune_Classification
    - HP_Tune_Regression

Instantiate the Pywedge_HP class & call the HP_Tune_CLassification method from Pywedge_HP class,

```python
pph = pw.Pywedge_HP(train, test, c, y, tracking=False)
pph.HP_Tune_Classification()
```

Args:
1) train = train dataframe
2) test = test dataframe
3) c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
4) y = target column name as a string 
5) tracking = True/False(Default) #to enable mlflow hyperpameter tracking

- For classification - HP_Tune_Classification() 
- For Regression - HP_Tune_Regression()

![HP_Tune](https://raw.githubusercontent.com/taknev83/pywedge/main/images/HP_tune.gif)  
    
As seen in the above GIF, user can interactively enter hyperparameter values, without worrying about tracking the same, as the integreted MLFlow automatically takes care of tracking hyperparameter values. 

To invoke mlflow tracking user interface, follow the below steps,
1) open command prompt
2) change directory to the location of the Jupyter Notebook file, for eg., if Jupyter notebook in a folder named pywedge in Documents folder, 
    ```
    cd documents\pywedge
    ```
3) enter the following command from the same folder as of the Jupyter Notebook file,
    ```
    mlflow ui
    ```
4) which trigers the mlflow ui & typically mlflow starts in the local host 5000. Please find the below pic for reference,

![mlflow_cmd](https://raw.githubusercontent.com/taknev83/pywedge/main/images/Mlflow_cmd.JPG)


Regression Hyperparameter tuning is in the same lines of above steps.

# [>> Pywedge Documentation](https://taknev83.github.io/pywedge-docs/)


### The following additions to pywedge is planned,
- [X] A separate method to produce good charts
- [ ] To handle NLP column
- [ ] To handle time series dataset
- [ ] To handle stock prices specific analysis





Requires Python 64 bit

THIS IS IN BETA VERSION 
