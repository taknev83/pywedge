# Pywedge Modules

## Make_Charts
Makes 8 different types of interactive Charts with interactive axis selection widgets in a single line of code for the given dataset.

***Different types of Charts viz,***

* Scatter Plot
* Pie Chart
* Bar Plot
* Violin Plot
* Box Plot
* Distribution Plot
* Histogram
* Correlation Plot

**Arguments:**

* Dataframe
* c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
* y = target column name as a string

**Returns:**

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

## Baseline_model
The baseline_model class starts with interactive pre-processing steps,
![baseline_model](https://raw.githubusercontent.com/taknev83/pywedge/main/images/baseline_models_inputs.jpg)

Instantiate the baseline class & call the classification_summary method from baseline_model class,

```python
blm = pw.baseline_model(train, test, c, y, type)
blm.classification_summary()
```

* Args:
    * train = train dataframe
    * test = test dataframe
    * c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
    * y = target column name as a string
    * type = Classification(Default) / Regression


- For classification - classification_summary()
- For Regression - Regression_summary()

**User Inputs in Interactive Tab:**

* Categorical columns conversion options
    *   Using Pandas Catcodes
    *   Using Pandas Get Dummies
* Standardization Options,
    *   Standard scalar
    *   Minmax scalar
    *   Robust Scalar
    *   No Standardization
* For Classification, Class balance using SMOTE options
    *   Yes
    *   No
* Test Size for Train-test split
    *   test size in float

**Returns:**

* Baseline models tab - Various baseline model metrics
* Predict Baseline model tab - User can select the preferred available baseline choices to predict

![baseline_output](https://raw.githubusercontent.com/taknev83/pywedge/main/images/baseline_model_output.gif)


## Pywedge_Interactive Hyperparameter Tuning

* Introducing interactive hyperparameter tuning classes, Pywedge_HP, which has following two methods,
    - HP_Tune_Classification
    - HP_Tune_Regression

Instantiate the Pywedge_HP class & call the HP_Tune_Classification method from Pywedge_HP class,

```python
pph = pw.Pywedge_HP(train, test, c, y, tracking=False)
pph.HP_Tune_Classification()
```

**Args:**

* train = train dataframe
* test = test dataframe
* c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
* y = target column name as a string
* tracking = True/False(Default) #to enable mlflow hyperpameter tracking

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
