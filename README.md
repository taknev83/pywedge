# pywedge
Cleans raw data, runs baseline models 

Cleans the raw dataframe to fed into ML models. Following data pre_processing will be carried out,
1) segregating numeric & categorical columns
2) missing values imputation for numeric & categorical columns
3) standardization
4) feature importance
5) SMOTE
6) baseline model

Pre_process_data()
Inputs: 
1) train = train dataframe
2) test = test dataframe
3) c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
4) y = target column name as a string 
5) type = Classification / Regression

Returns:
1) new_X (cleaned feature columns in dataframe)
2) new_y (cleaned target column in dataframe)  
3) new_test (cleaned stand out test dataset)

baseline_model()
- For classification - classification_summary() 
- For Regression - Regression_summary()

Inputs:
1) new_x
2) new_y

Returns:
Various baseline model metrics 
