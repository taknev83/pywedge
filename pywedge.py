
class Pre_process_data():
    ''' Cleans the raw dataframe to fed into ML models. Following data pre_processing will be carried out,
        1) segregating numeric & categorical columns
        2) missing values imputation for numeric & categorical columns
        3) standardization
        4) feature importance
        5) SMOTE
        6) baseline model

        Inputs: 
        1) train = train dataframe
        2) test = stand out test dataframe (without target column)
        2) c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
        3) y = target column name as a string 
        4) type = Classification / Regression

        Returns:
        1) new_X (cleaned feature columns in dataframe)
        2) new_y (cleaned target column in dataframe)
        3) new_test (cleaned stand out test dataframe

    '''
    import pandas as pd
    
    def __init__(self, train, test, c, y, type="Classification"):
        self.train = train
        self.test = test
        self.c = c
        self.y = y
        self.X = self.train.drop(self.y,1)
        self.type = type
                       

    def missing_zero_values_table(self):
        import pandas as pd
        zero_val = (self.train == 0.00).astype(int).sum(axis=0)
        mis_val = self.train.isnull().sum()
        mis_val_percent = 100 * self.train.isnull().sum() / len(self.train)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(self.train)
        mz_table['Data Type'] = self.train.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected train dataframe has " + str(self.train.shape[1]) + " columns and " + str(self.train.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
        print(mz_table)

    
    def feature_importance(self):
        if self.type=="Classification":
            from sklearn.ensemble import AdaBoostClassifier
            import warnings
            import pandas as pd
            warnings.filterwarnings('ignore')
            ab = AdaBoostClassifier().fit(self.X, self.y)
            print(pd.Series(ab.feature_importances_, index=self.X.columns).sort_values(ascending=False).head(10))

        else:
            from sklearn.ensemble import AdaBoostRegressor
            import warnings
            warnings.filterwarnings('ignore')
            ab = AdaBoostRegressor().fit(self.X, self.y)
            print(pd.Series(ab.feature_importances_, index=self.X.columns).sort_values(ascending=False).head(10))

    
    def dataframe_clean(self):
        import pandas as pd
        train1 = self.train.copy(deep=True)
        print('Reading the datasets...')
        print('******************************************\n')
        if self.y!=None:
            self.X=self.train.drop(self.y,1)
        else:
            pass
        
        print('Train Dataframe summary...')
        print('******************************************\n')
        target = pd.DataFrame(self.train[[self.y]])

        self.missing_zero_values_table()
        print('******************************************')
        
        if self.type=="Classification":
            print('Class balance summary table')
            cb = self.train[self.y].value_counts()
            print(cb)
            cb_per = (self.train[self.y].value_counts()[1])/(self.train[self.y].value_counts()[0]+self.train[self.y].value_counts()[1])
            print('\n Class imbalance % is ', round(cb_per*100,2), '%')
        else:
            pass
        
        print('Starting data cleaning...')
        print('******************************************')
        if self.c!=None:
            self.X=self.train.drop([self.c,self.y],1)
            self.test=self.test.drop([self.c],1)
        else:
            pass
        
        categorical_cols = self.X.select_dtypes('object').columns.to_list()
        for col in categorical_cols:
            self.X[col].fillna(self.X[col].mode()[0], inplace=True)
        numeric_cols = self.X.select_dtypes(['float64', 'int64']).columns.to_list()
        for col in numeric_cols:
            self.X[col].fillna(self.X[col].mean(), inplace=True)
        cat_info = input('Do you want to use get_dummies or catcodes to convert categorical to numerical? \n\tpress 1 for catcodes - Quick info link - https://bit.ly/3lruqtf \n\tpress 2 for getdummies - Quick info link - https://bit.ly/3d76p7A \n')
        if cat_info == '1':
            for col in categorical_cols:
                self.X[col] = self.X[col].astype('category')
                self.X[col] = self.X[col].cat.codes
        else:        
            self.X = pd.get_dummies(self.X,drop_first=True)
     
        target = pd.get_dummies(target,drop_first=True)
        
        test_categorical_cols = self.test.select_dtypes('object').columns.to_list()

        for col in test_categorical_cols:
            self.test[col].fillna(self.test[col].mode()[0], inplace=True)
        numeric_cols = self.test.select_dtypes(['float64', 'int64']).columns.to_list()
        for col in numeric_cols:
            self.test[col].fillna(self.test[col].mean(), inplace=True)

        if cat_info == '1':
            for col in categorical_cols:
                self.test[col] = self.test[col].astype('category')
                self.test[col] = self.test[col].cat.codes
        else:        
            self.test = pd.get_dummies(self.test,drop_first=True)
        print('Comleted categorical column transformation')
       
        print('******************************************')
        std_scr = input('Do you want to standardize the data? \n\tpress 1 for Standard Scalar - Quick info link - https://bit.ly/2GPyG6w \n\tpress 2 for Robust Scalar - Quick info link - https://bit.ly/3jFNCD5 \n\tpress 3 for MinMax Scalar - Quick info link - https://bit.ly/2GKYJvX \n\tpress n for no standardize\n')
        if std_scr == '1':
            from sklearn.preprocessing import StandardScaler
            scalar = StandardScaler()
            scaled_df = pd.DataFrame(scalar.fit_transform(self.X), columns=self.X.columns, index=self.X.index)
            scaled_test = pd.DataFrame(scalar.fit_transform(self.test), columns=self.test.columns, index=self.test.index)
            print('standardization using Standard Scalar completed')
        elif std_scr == '2':
            from sklearn.preprocessing import RobustScaler
            scalar = RobustScaler()
            scaled_df = pd.DataFrame(scalar.fit_transform(self.X), columns=self.X.columns, index=self.X.index)
            scaled_test = pd.DataFrame(scalar.fit_transform(self.test), columns=self.test.columns, index=self.test.index)
            print('standardization using Roubust Scalar completed')
            
        elif std_scr == '3':
            from sklearn.preprocessing import MinMaxScaler
            scalar = MinMaxScaler()
            scaled_df = pd.DataFrame(scalar.fit_transform(self.X), columns=self.X.columns, index=self.X.index)
            scaled_test = pd.DataFrame(scalar.fit_transform(self.test), columns=self.test.columns, index=self.test.index)
            print('standardization using Minmax Scalar completed')

        else:
            pass

        if std_scr!='n':
            scaled_x = scaled_df
            scaled_test = scaled_test
        else:
            scaled_x = self.X
            scaled_test = self.test
            
        if self.type=="Classification":

            apply_smote = input('Do you want to apply SMOTE to oversample the minority class? y/n \n\tQuick info on SMOTE - https://bit.ly/3nlGCNX \n')
            if apply_smote == 'y':
                from imblearn.over_sampling import SMOTE
                import warnings
                warnings.filterwarnings('ignore', 'FutureWarning') 
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(scaled_x, target)
                X_res = pd.DataFrame(X_res, columns=scaled_x.columns)
                y_res = pd.DataFrame(y_res, columns=target.columns)
                print('Oversampling using SMOTE completed')
              
            else:
                print('No oversampling done')
                X_res = scaled_x
                y_res = target
        else:
            X_res = scaled_x
            y_res = target
        
        return X_res, y_res, scaled_test
    
    

class baseline_model():

    '''
    Feed the input from cleaned feature & target dataframes to quickly get a summary of various classification/regressions model performance. 

    Inputs:
    1) X = Cleaned features dataframe  
    2) y = Cleaned target dataframe

    Returns:
    Various classification/regressions models & model performances

    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y
 
    def classification_summary(self):
        print('Starting classification_summary...')
        print('TOP 10 FEATURE IMPORTANCE')
        from sklearn.ensemble import AdaBoostClassifier
        import pandas as pd
        import warnings
        warnings.filterwarnings('ignore')
        ab = AdaBoostClassifier().fit(self.X, self.y)
        print(pd.Series(ab.feature_importances_, index=self.X.columns).sort_values(ascending=False).head(10))

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
                    self.X.values, self.y.values, test_size=float(input("Please enter test size (for eg. please enter 0.20 for 20% test size):\n")), random_state=1)
             
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from catboost import CatBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.metrics import roc_auc_score
        import warnings
        warnings.filterwarnings('ignore')
        classifiers = {
            "Logistic"     : LogisticRegression(max_iter=1000),
            "KNN(3)"       : KNeighborsClassifier(3), 
            "Decision Tree": DecisionTreeClassifier(max_depth=7), 
            "Random Forest": RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4), 
            "Neural Net"   : MLPClassifier(alpha=1), 
            "XGBoost"      : xgb.XGBClassifier(max_depth=4, n_estimators=10, learning_rate=0.1, n_jobs=1),
            "AdaBoost"     : AdaBoostClassifier(),
            "CatBoost"     : CatBoostClassifier(silent=True),
            "Naive Bayes"  : GaussianNB(), 
            "QDA"          : QuadraticDiscriminantAnalysis(),
            "Linear SVC"   : LinearSVC(),
            "Linear SVM"   : SVC(kernel="linear"), 
            "Gaussian Proc": GaussianProcessClassifier(1.0 * RBF(1.0)),
        }
        from time import time
        k = 10      
        head = list(classifiers.items())[:k]

        for name, classifier in head:
            start = time()
            classifier.fit(X_train, y_train)
            train_time = time() - start
            start = time()
            predictions = classifier.predict(X_test)
            predict_time = time()-start
            acc_score= (accuracy_score(y_test,predictions))
            roc_score= (roc_auc_score(y_test,predictions))
            f1_macro= (f1_score(y_test, predictions, average='macro'))
            print("{:<15}| acc_score = {:.3f} | roc_score = {:,.3f} | f1_score(macro) = {:,.3f} | Training time = {:,.3f} | Pred. time = {:,.3f}".format(name, acc_score, roc_score, f1_macro, train_time, predict_time))
        
    def Regression_summary(self):
        print('Starting regression summary...')
        print('TOP 10 FEATURE IMPORTANCE TABLE')
        from sklearn.ensemble import AdaBoostRegressor
        import pandas as pd
        import warnings
        warnings.filterwarnings('ignore')
        ab = AdaBoostRegressor().fit(self.X, self.y)
        print(pd.Series(ab.feature_importances_, index=self.X.columns).sort_values(ascending=False).head(10))

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
                    self.X.values, self.y.values, test_size=float(input("Please enter test size: (for eg. please enter 0.20 for 20% test size): \n\t")), random_state=1)
        from time import time
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from sklearn.svm import LinearSVR
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.metrics import explained_variance_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
        from catboost import CatBoostRegressor
        from sklearn.neural_network import MLPRegressor
        import xgboost as xgb
        from math import sqrt
        import warnings
        warnings.filterwarnings('ignore')
        print('--------------------------LINEAR MODELS---------------------------------')
        lin_regressors = {
            'Linear Reg'       : LinearRegression(), 
            'KNN'              : KNeighborsRegressor(),
            'SVR'              : SVR(),
            'LinearSVR'        : LinearSVR(),
            'Lasso'            : Lasso(),
            'Ridge'            : Ridge(),
        }

        from time import time
        k = 10      
        head = list(lin_regressors.items())[:k]

        for name, lin_regressors in head:
            start = time()
            lin_regressors.fit(X_train, y_train)
            train_time = time() - start
            start = time()
            predictions = lin_regressors.predict(X_test)
            predict_time = time()-start
            exp_var = explained_variance_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_absolute_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            print("{:<15}| exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} | Train time = {:,.3f} | Pred. time = {:,.3f}".format(name, exp_var, mae, rmse, r2, train_time, predict_time))
        
        print('------------------------NON LINEAR MODELS----------------------------------')
        print('---------------------THIS MIGHT TAKE A WHILE-------------------------------')
        non_lin_regressors = {
            'SVR'           : SVR(),  
            'Decision Tree' : DecisionTreeRegressor(max_depth=5),
            'Random Forest' : RandomForestRegressor(max_depth=10),
            'GB Regressor'  : GradientBoostingRegressor(n_estimators=200),
            'CB Regressor'  : CatBoostRegressor(silent=True),
            'ADAB Regressor': AdaBoostRegressor(),
            'MLP Regressor' : MLPRegressor(),
            'XGB Regressor' : xgb.XGBRegressor()
        }

        
        from time import time
        k = 10      
        head = list(non_lin_regressors.items())[:k]

        for name, non_lin_regressors in head:
            start = time()
            non_lin_regressors.fit(X_train, y_train)
            train_time = time() - start
            start = time()
            predictions = non_lin_regressors.predict(X_test)
            predict_time = time()-start
            exp_var = explained_variance_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_absolute_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            print("{:<15}| exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} | Train time = {:,.3f} | Pred. time = {:,.3f}".format(name, exp_var, mae, rmse, r2, train_time, predict_time))
        
