class Pywedge_Charts():
    '''
    Makes 8 different types of interactive Charts with interactive axis selection widgets in a single line of code for the given dataset. 
    Different types of Charts viz,
        1. Scatter Plot
        2. Pie Chart
        3. Bar Plot
        4. Violin Plot
        5. Box Plot
        6. Distribution Plot
        7. Histogram 
        8. Correlation Plot
    
    Inputs:
        1. Dataframe
        2. c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
        3. y = target column name as a string 
        
    Returns:
        Charts widget
    '''
   
    def __init__(self, train, c, y, manual=True):
        self.train = train
        self.c = c
        self.y = y
        self.X = self.train.drop(self.y,1)
        self.manual = manual
 
    def make_charts(self): 
        import pandas as pd
        import ipywidgets as widgets
        import plotly.express as px
        import plotly.figure_factory as ff
        import plotly.offline as pyo
        from ipywidgets import HBox, VBox, Button
        from ipywidgets import  interact, interact_manual, interactive
        import plotly.graph_objects as go
        from plotly.offline import iplot
        
        header = widgets.HTML(value="<h2>Pywedge Make_Charts </h2>")
        display(header)

        
        if len(self.train) > 500:
            from sklearn.model_selection import train_test_split
            test_size = 500/len(self.train)
            if self.c!=None:
                data = self.X.drop(self.c,1)
            else:
                data = self.X
                
            target = self.train[self.y]
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=1)
            train_mc = pd.concat([X_test, y_test], axis=1)
        else:
            train_mc = self.train

        train_numeric = train_mc.select_dtypes('number')
        train_cat = train_mc.select_dtypes(exclude='number')

        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out8 = widgets.Output()
        
        tab = widgets.Tab(children = [out1, out2, out3, out4, out5, out6, out7, out8])
        tab.set_title(0, 'Scatter Plot')
        tab.set_title(1, 'Pie Chart')
        tab.set_title(2, 'Bar Plot')
        tab.set_title(3, 'Violin Plot')  
        tab.set_title(4, 'Box Plot')
        tab.set_title(5, 'Distribution Plot')
        tab.set_title(6, 'Histogram')
        tab.set_title(7, 'Correlation plot')
        
        display(tab)
        
        with out1:
            header = widgets.HTML(value="<h1>Scatter Plots </h1>")
            display(header)

            x = widgets.Dropdown(options=list(train_mc.select_dtypes('number').columns))
            def scatter_plot(X_Axis=list(train_mc.select_dtypes('number').columns), 
                            Y_Axis=list(train_mc.select_dtypes('number').columns)[1:], 
                            Color=list(train_mc.select_dtypes('number').columns)):
        
                fig = go.FigureWidget(data=go.Scatter(x=train_mc[X_Axis],
                                                y=train_mc[Y_Axis],
                                                mode='markers',
                                                text=list(train_cat),
                                                marker_color=train_mc[Color]))
        
                fig.update_layout(title=f'{Y_Axis.title()} vs {X_Axis.title()}', 
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}', 
                                autosize=False,width=600,height=600)
                fig.show()
        
            widgets.interact_manual.opts['manual_name'] = 'Make_Chart'
            one = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(scatter_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)
            
        with out2:
            header = widgets.HTML(value="<h1>Pie Charts </h1>")
            display(header)

            def pie_chart(Labels=list(train_mc.select_dtypes(exclude='number').columns),
                        Values=list(train_mc.select_dtypes('number').columns)[0:]):
                
                fig = go.FigureWidget(data=[go.Pie(labels=train_mc[Labels], values=train_mc[Values])])
                
                fig.update_layout(title=f'{Values.title()} vs {Labels.title()}',
                                autosize=False,width=500,height=500)
                fig.show()    
            one = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(pie_chart, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)

        with out3:
            header = widgets.HTML(value="<h1>Bar Plots </h1>")
            display(header)

            def bar_plot(X_Axis=list(train_mc.select_dtypes(exclude='number').columns), 
                        Y_Axis=list(train_mc.select_dtypes('number').columns)[1:],
                        Color=list(train_mc.select_dtypes(exclude='number').columns)):

                fig1 = px.bar(train_mc, x=train_mc[X_Axis], y=train_mc[Y_Axis], color=train_mc[Color])       
                fig1.update_layout(barmode='group', 
                                title=f'{X_Axis.title()} vs {Y_Axis.title()}', 
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig1.show()
            one = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(bar_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)

                
        with out4:
            header = widgets.HTML(value="<h1>Violin Plots </h1>")
            display(header)

            def viol_plot(X_Axis=list(train_mc.select_dtypes('number').columns), 
                          Y_Axis=list(train_mc.select_dtypes('number').columns)[1:], 
                          Color=list(train_mc.select_dtypes(exclude='number').columns)):
                
                fig2 = px.violin(train_mc, X_Axis, Y_Axis, Color, box=True, hover_data=train_mc.columns)
                fig2.update_layout(title=f'{X_Axis.title()} vs {Y_Axis.title()}', 
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()
        
            one = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(viol_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)
        
                
        with out5:
            header = widgets.HTML(value="<h1>Box Plots </h1>")
            display(header)
            
            def box_plot(X_Axis=list(train_mc.select_dtypes(exclude='number').columns),
                        Y_Axis=list(train_mc.select_dtypes('number').columns)[0:],
                        Color=list(train_mc.select_dtypes(exclude='number').columns)):


                fig4 = px.box(train_mc, x=X_Axis, y=Y_Axis, color=Color, points="all")
 
                fig4.update_layout(barmode='group', 
                                title=f'{X_Axis.title()} vs {Y_Axis.title()}', 
                                xaxis_title=f'{X_Axis.title()}',
                                yaxis_title=f'{Y_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig4.show()
                
            one = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(box_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)

        with out6:
            header = widgets.HTML(value="<h1>Distribution Plots </h1>")
            display(header)

            def dist_plot(X_Axis=list(train_mc.select_dtypes('number').columns), 
                          Y_Axis=list(train_mc.select_dtypes('number').columns)[1:], 
                          Color=list(train_mc.select_dtypes(exclude='number').columns)):
                
                fig2 = px.histogram(train_mc, X_Axis, Y_Axis, Color, marginal='violin', hover_data=train_mc.columns)
                fig2.update_layout(title=f'{X_Axis.title()} vs {Y_Axis.title()}', 
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()
        
            one = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(dist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)       
        
        with out7:
            header = widgets.HTML(value="<h1>Histogram </h1>")
            display(header)

            def hist_plot(X_Axis=list(train_mc.columns)):
                fig2 = px.histogram(train_mc, X_Axis)
                fig2.update_layout(title=f'{X_Axis.title()}', 
                                xaxis_title=f'{X_Axis.title()}',
                                autosize=False,width=600,height=600)
                fig2.show()
                    
        
            one = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            two = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            three = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})
            four = interactive(hist_plot, {'manual': self.manual, 'manual_name':'Make_Chart'})

            g = widgets.HBox([one, two])
            display(g)
            h = widgets.HBox([three, four])
            display(h)

        with out8:

            header = widgets.HTML(value="<h1>Correlation Plots </h1>")
            display(header)

            import plotly.figure_factory as ff
            corrs = train_mc.corr()
            colorscale = ['Greys', 'Greens', 'Bluered', 'RdBu',
                    'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
                    'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
            @interact_manual
            def plot_corrs(colorscale=colorscale):
                figure = ff.create_annotated_heatmap(z = corrs.round(2).values, 
                                                x =list(corrs.columns), 
                                                y=list(corrs.index), 
                                                colorscale=colorscale,
                                                annotation_text=corrs.round(2).values)
                iplot(figure)

class baseline_model():

    ''' 
        Cleans the raw dataframe to fed into ML models and runs various baseline models. Following data pre_processing will be carried out,
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
        4) arious classification/regressions models & model performances
    '''
                 
                 
    def __init__(self, train, test, c, y, type="Classification"):
        self.train = train
        self.test = test
        self.c = c
        self.y = y
        self.type = type
        self.X = train.drop(self.y,1)
 
    def classification_summary(self):
        import ipywidgets as widgets
        from ipywidgets import HBox, VBox, Button
        from IPython.display import display, Markdown, clear_output

        header = widgets.HTML(value="<h2>Pywedge Baseline Models </h2>")
        display(header)

        out1 = widgets.Output()
        out2 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2])
        tab.set_title(0,'Baseline Models')
        tab.set_title(1, 'Predict Baseline Model')
        display(tab)


        with out1:
            import ipywidgets as widgets
            from ipywidgets import HBox, VBox, Button
            from IPython.display import display, Markdown, clear_output

            header = widgets.HTML(value="<h2>Pre_processing </h2>")
            display(header)

            import pandas as pd
            cat_info = widgets.Dropdown(
                options = [('cat_codes', '1'), ('get_dummies', '2')],
                value = '1',
                description = 'Select categorical conversion',
                style = {'description_width': 'initial'},     
                disabled=False)

            std_scr = widgets.Dropdown(
                options = [('StandardScalar', '1'), ('RobustScalar', '2'), ('MinMaxScalar', '3'), ('No Standardization', 'n')],
                value = 'n',
                description = 'Select Standardization methods',
                style = {'description_width': 'initial'},     
                disabled=False)

            apply_smote = widgets.Dropdown(
                options = [('Yes', 'y'), ('No', 'n')],
                value = 'y',
                description = 'Do you want to apply SMOTE?',
                style = {'description_width': 'initial'},     
                disabled=False)

            pp_class = widgets.VBox([cat_info, std_scr, apply_smote])
            pp_reg = widgets.VBox([cat_info, std_scr])
            if self.type == 'Classification':
                display(pp_class)
            else:
                display(pp_reg)

            test_size = widgets.BoundedFloatText(
                value=0.20,
                min=0.05,
                max=0.5,
                step=0.05,
                description='Text Size %',
                disabled=False)
            
            display(test_size)

            button_1 = widgets.Button(description = 'Run Baseline models')
            out = widgets.Output()

            def on_button_clicked(_):
                with out:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
        
                    if self.type=="Classification":
                        if apply_smote.value == 'y':
                            from imblearn.over_sampling import SMOTE
                            import warnings
                            warnings.simplefilter(action='ignore', category=FutureWarning)
                            from sklearn.exceptions import DataConversionWarning
                            warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                            warnings.filterwarnings('ignore', 'FutureWarning') 
                            sm = SMOTE(random_state=42)
                            self.new_X, self.new_y= sm.fit_resample(self.new_X, self.new_y)
                            self.new_X = pd.DataFrame(self.new_X, columns=self.new_X.columns)
                            self.new_y= pd.DataFrame(self.new_y, columns=self.new_y.columns)
                            print('> Oversampling using SMOTE completed')
                        
                        else:
                            print('> No oversampling done')

                    print('\nStarting classification_summary...')
                    print('TOP 10 FEATURE IMPORTANCE - USING ADABOOST CLASSIFIER')
                    from sklearn.ensemble import AdaBoostClassifier
                    import pandas as pd
                    import warnings
                    warnings.filterwarnings('ignore')
                    ab = AdaBoostClassifier().fit(self.new_X, self.new_y)
                    print(pd.Series(ab.feature_importances_, index=self.new_X.columns).sort_values(ascending=False).head(10))

                    from sklearn.model_selection import train_test_split
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                self.new_X.values, self.new_y.values, test_size=test_size.value, random_state=1)
                        
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
                        "MLP Classiz"   : MLPClassifier(alpha=1), 
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
                        classifier.fit(self.X_train, self.y_train)
                        train_time = time() - start
                        start = time()
                        predictions = classifier.predict(self.X_test)
                        predict_time = time()-start
                        acc_score= (accuracy_score(self.y_test,predictions))
                        roc_score= (roc_auc_score(self.y_test,predictions))
                        f1_macro= (f1_score(self.y_test, predictions, average='macro'))
                        print("{:<15}| acc_score = {:.3f} | roc_score = {:,.3f} | f1_score(macro) = {:,.3f} | Train time = {:,.3f}s | Pred. time = {:,.3f}s".format(name, acc_score, roc_score, f1_macro, train_time, predict_time))

            button_1.on_click(on_button_clicked)

            a = widgets.VBox([button_1, out])
            display(a)
            
            with out2:
                base_model = widgets.Dropdown(
                options=['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'MLP Classifier', 'XGBoost', 'AdaBoost', 'CatBoost'],
                value='Logistic Regression',
                description='Choose Base Model: ',
                style = {'description_width': 'initial'},     
                disabled=False)
                
                display(base_model)
                
                button_2 = widgets.Button(description = 'Predict Baseline models')
                out2 = widgets.Output()

                def on_pred_button_clicked(_):
                    with out2:
                        from sklearn.neural_network import MLPClassifier
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
                        from catboost import CatBoostClassifier
                        from sklearn.linear_model import LogisticRegression
                        import xgboost as xgb
                        clear_output()
                        print(base_model.value)

                        if base_model.value == 'Logistic Regression':
                            classifier = LogisticRegression(max_iter=1000)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('> Prediction completed. \n> Use dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'KNN':
                            classifier = KNeighborsClassifier(3)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'Decision Tree':
                            classifier = DecisionTreeClassifier(max_depth=7)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'Random Forest':
                            classifier = RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'MLP Classifier':
                            classifier = MLPClassifier(alpha=1)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'XGBoost':
                            classifier = xgb.XGBClassifier(max_depth=4, n_estimators=10, learning_rate=0.1, n_jobs=1)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'AdaBoost':
                            classifier = AdaBoostClassifier()
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'CatBoost':
                            classifier = CatBoostClassifier(silent=True)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        
                button_2.on_click(on_pred_button_clicked)
    
                b = widgets.VBox([button_2, out2])
                display(b)

    def Regression_summary(self):
        import ipywidgets as widgets
        from ipywidgets import HBox, VBox, Button
        from IPython.display import display, Markdown, clear_output

        header = widgets.HTML(value="<h2>Pywedge Baseline Models </h2>")
        display(header)

        out1 = widgets.Output()
        out2 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2])
        tab.set_title(0,'Baseline Models')
        tab.set_title(1, 'Predict Baseline Model')
        display(tab)

        with out1:                          
            import ipywidgets as widgets
            from ipywidgets import HBox, VBox, Button
            from IPython.display import display, Markdown, clear_output

            header = widgets.HTML(value="<h2>Pre_processing </h2>")
            display(header)

            import pandas as pd
            cat_info = widgets.Dropdown(
                options = [('cat_codes', '1'), ('get_dummies', '2')],
                value = '1',
                description = 'Select categorical conversion',
                style = {'description_width': 'initial'},     
                disabled=False)

            std_scr = widgets.Dropdown(
                options = [('StandardScalar', '1'), ('RobustScalar', '2'), ('MinMaxScalar', '3'), ('No Standardization', 'n')],
                value = 'n',
                description = 'Select Standardization methods',
                style = {'description_width': 'initial'},     
                disabled=False)

            apply_smote = widgets.Dropdown(
                options = [('Yes', 'y'), ('No', 'n')],
                value = 'y',
                description = 'Do you want to apply SMOTE?',
                style = {'description_width': 'initial'},     
                disabled=False)

            pp_class = widgets.VBox([cat_info, std_scr, apply_smote])
            pp_reg = widgets.VBox([cat_info, std_scr])
            if self.type == 'Classification':
                display(pp_class)
            else:
                display(pp_reg)

            test_size = widgets.BoundedFloatText(
                value=0.20,
                min=0.05,
                max=0.5,
                step=0.05,
                description='Text Size %',
                disabled=False)
            
            display(test_size)

            button_1 = widgets.Button(description = 'Run Baseline models')
            out = widgets.Output()

            def on_button_clicked(_):
                with out:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
        
                    if self.type=="Classification":
                        if apply_smote.value == 'y':
                            from imblearn.over_sampling import SMOTE
                            import warnings
                            warnings.simplefilter(action='ignore', category=FutureWarning)
                            from sklearn.exceptions import DataConversionWarning
                            warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                            warnings.filterwarnings('ignore', 'FutureWarning') 
                            sm = SMOTE(random_state=42)
                            self.new_X, self.new_y= sm.fit_resample(self.new_X, self.new_y)
                            self.new_X = pd.DataFrame(self.new_X, columns=self.new_X.columns)
                            self.new_y= pd.DataFrame(self.new_y, columns=self.new_y.columns)
                            print('> Oversampling using SMOTE completed')
                        
                        else:
                            print('> No oversampling done')

                    print('Starting regression summary...')
                    print('TOP 10 FEATURE IMPORTANCE TABLE')
                    from sklearn.ensemble import AdaBoostRegressor
                    import pandas as pd
                    import warnings
                    warnings.filterwarnings('ignore')
                    ab = AdaBoostRegressor().fit(self.new_X, self.new_y)
                    print(pd.Series(ab.feature_importances_, index=self.new_X.columns).sort_values(ascending=False).head(10))


                    from sklearn.model_selection import train_test_split
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                self.new_X.values, self.new_y.values, test_size=test_size.value, random_state=1)
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
                        'LinearSVR'        : LinearSVR(),
                        'Lasso'            : Lasso(),
                        'Ridge'            : Ridge(),
                    }

                    from time import time
                    k = 10      
                    head = list(lin_regressors.items())[:k]

                    for name, lin_regressors in head:
                        start = time()
                        lin_regressors.fit(self.X_train, self.y_train)
                        train_time = time() - start
                        start = time()
                        predictions = lin_regressors.predict(self.X_test)
                        predict_time = time()-start
                        exp_var = explained_variance_score(self.y_test, predictions)
                        mae = mean_absolute_error(self.y_test, predictions)
                        rmse = sqrt(mean_absolute_error(self.y_test, predictions))
                        r2 = r2_score(self.y_test, predictions)
                        
                        print("{:<15}| exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} | Train time = {:,.3f}s | Pred. time = {:,.3f}s".format(name, exp_var, mae, rmse, r2, train_time, predict_time))
                    
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
                        non_lin_regressors.fit(self.X_train, self.y_train)
                        train_time = time() - start
                        start = time()
                        predictions = non_lin_regressors.predict(self.X_test)
                        predict_time = time()-start
                        exp_var = explained_variance_score(self.y_test, predictions)
                        mae = mean_absolute_error(self.y_test, predictions)
                        rmse = sqrt(mean_absolute_error(self.y_test, predictions))
                        r2 = r2_score(self.y_test, predictions)
                        
                        print("{:<15}| exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} | Train time = {:,.3f}s | Pred. time = {:,.3f}s".format(name, exp_var, mae, rmse, r2, train_time, predict_time))

            button_1.on_click(on_button_clicked)

            a = widgets.VBox([button_1, out])
            display(a)

            with out2:
                base_model = widgets.Dropdown(
                options=['Linear Regression', 'KNN', 'Decision Tree', 'Random Forest', 'MLP Regressor', 'XGBoost', 'AdaBoost', 'CatBoost'],
                value='Linear Regression',
                description='Choose Base Model: ',
                style = {'description_width': 'initial'},     
                disabled=False)
                
                display(base_model)
                
                button_2 = widgets.Button(description = 'Predict Baseline models')
                out2 = widgets.Output()

                def on_pred_button_clicked(_):
                    with out2:
                        from sklearn.neighbors import KNeighborsRegressor
                        from sklearn.linear_model import LinearRegression
                        from sklearn.tree import DecisionTreeRegressor
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
                        from catboost import CatBoostRegressor
                        from sklearn.neural_network import MLPRegressor
                        import xgboost as xgb
                        clear_output()
                        print(base_model.value)

                        if base_model.value == 'Linear Regression':
                            classifier = LinearRegression()
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'KNN':
                            classifier = KNeighborsRegressor()
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'Decision Tree':
                            classifier = DecisionTreeRegressor(max_depth=5)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'Random Forest':
                            classifier = RandomForestRegressor(max_depth=10)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'MLP Regressor':
                            classifier = MLPRegressor()
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'XGBoost':
                            classifier = xgb.XGBRegressor()
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'AdaBoost':
                            classifier = AdaBoostRegressor()
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        if base_model.value == 'CatBoost':
                            classifier = CatBoostRegressor(silent=True)
                            classifier.fit(self.X_train, self.y_train)
                            self.predictions_baseline = classifier.predict(self.new_test)
                            print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., blm.predictions_baseline, where blm is pywedge_baseline_model class object')
                        
                button_2.on_click(on_pred_button_clicked)
    
                b = widgets.VBox([button_2, out2])
                display(b)
                
class Pywedge_HP():
    '''
    Creates interative widget based Hyperparameter selection tool for both Classification & Regression.
    
    For Classification, following baseline estimators are covered in Gridsearch & Randomized search options
    1) Logistic Regression
    2) Decision Tree
    3) Random Forest
    4) KNN Classifier

    For Regression, following baseline estimators are covered in Gridsearch & Randomized search options
    1) Linear Regression
    2) Decision Tree Regressor
    3) Random Forest Regressor
    4) KNN Regressor

    Inputs:
        1) train = train dataframe
        2) test = stand out test dataframe (without target column)
        3) c = any redundant column to be removed (like ID column etc., at present supports a single column removal, subsequent version will provision multiple column removal requirements)
        4) y = target column name as a string         
         
    Ouputs:
        1) Hyperparameter results
        2) Predict on standout test dataset
    '''
    
    def __init__(self, train, test, c, y):
        self.train = train
        self.test = test
        self.c = c
        self.y = y
        self.X = train.drop(self.y,1)
        
    def HP_Tune_Classification(self):
        from sklearn.linear_model import LogisticRegression
        import mlflow
        from mlflow import log_metric, log_param, log_artifacts
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        import ipywidgets as widgets
        from ipywidgets import HBox, VBox, Button, Label
        from ipywidgets import  interact_manual, interactive, interact
        import logging
        from IPython.display import display, Markdown, clear_output
        import warnings
        warnings.filterwarnings('ignore')

        out1 = widgets.Output()
        out2 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2])
        tab.set_title(0, 'Input')
        tab.set_title(1, 'Output')

        display(tab)

        with out1:
            header_1 = widgets.HTML(value="<h2>Pywedge HP_Tune</h2>")
            header = widgets.HTML(value="<h3>Base Estimator</h3>")
            display(header_1, header)
            
            import pandas as pd
            cat_info = widgets.Dropdown(
                options = [('cat_codes', '1'), ('get_dummies', '2')],
                value = '1',
                description = 'Select categorical conversion',
                style = {'description_width': 'initial'},     
                disabled=False)

            std_scr = widgets.Dropdown(
                options = [('StandardScalar', '1'), ('RobustScalar', '2'), ('MinMaxScalar', '3'), ('No Standardization', 'n')],
                value = 'n',
                description = 'Select Standardization methods',
                style = {'description_width': 'initial'},     
                disabled=False)

            apply_smote = widgets.Dropdown(
                options = [('Yes', 'y'), ('No', 'n')],
                value = 'y',
                description = 'Do you want to apply SMOTE?',
                style = {'description_width': 'initial'},     
                disabled=False)

            pp_class = widgets.HBox([cat_info, std_scr, apply_smote])

            header_2 = widgets.HTML(value="<h3>Pre_processing </h3>")

            base_estimator = widgets.Dropdown(
            options=['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN Classifier'],
            value='Logistic Regression',
            description='Choose Base Estimator: ',
            style = {'description_width': 'initial'},     
            disabled=False)

            display(base_estimator)


            button = widgets.Button(description='Select Base Estimator')
            out = widgets.Output()

            def on_button_clicked(_):
                with out:
                    clear_output()
                    selection = base_estimator.value
                    if selection == 'Logistic Regression':
                        h1 = widgets.HTML('<h3>Select Logistic Regression Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(aa)
                    elif selection =='Decision Tree':
                        h1 = widgets.HTML('<h3>Select Decision Tree Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(frame)
                    elif selection == 'Random Forest':
                        h1 = widgets.HTML('<h3>Select Random Forest Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(frame_RF)
                    elif selection == 'KNN Classifier':
                        h1 = widgets.HTML('<h3>Select KNN Classifier Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(frame_K)

            button.on_click(on_button_clicked)

            a = widgets.VBox([button, out])
            display(a)

        # Logistic Regression Hyperparameters _Start

            penalty_L = widgets.SelectMultiple(
                options = ['l1', 'l2', 'elasticnet', 'none'],
                value = ['none'],
                rows = 4,
                description = 'Penalty',
                disabled = False)

            dual_L = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                rows = 2,       
                description = 'Dual',
                disabled = False)

            tol_L = widgets.Text(
                value='0.0001',
                placeholder='enter any float value',
                description='Tolerence (tol)',
                style = {'description_width': 'initial'},
                disabled=False)

            g = widgets.HBox([penalty_L, dual_L, tol_L])

            C_L = widgets.Text(
                value='1.0',
                placeholder='enter any float value',
                description='C',
                disabled=False)

            fit_intercept_L = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                rows = 2,
                description = 'Fit_intercept',
                disabled = False)

            intercept_scaling_L = widgets.Text(
                value='1.0',
                placeholder='enter any float value',
                description='Intercept_scaling',
                style = {'description_width': 'initial'},
                disabled=False)

            h = widgets.HBox([C_L, fit_intercept_L, intercept_scaling_L])

            class_weight_L = widgets.SelectMultiple(
                options = ['balanced', 'None'],
                value = ['None'],
                rows = 2,
                description = 'Class_weight',
                disabled = False)

            random_state_L = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Random_state',
                style = {'description_width': 'initial'},
                disabled=False)

            solver_L = widgets.SelectMultiple(
                options = ['newton-cg', 'lbfgs', 'sag', 'saga'],
                value = ['lbfgs'],
                rows = 4,
                description = 'Solver',
                disabled = False)

            i= widgets.HBox([class_weight_L, random_state_L, solver_L])

            max_iter_L = widgets.Text(
                value='100',
                placeholder='enter any integer value',
                description='Max_Iterations',
                style = {'description_width': 'initial'},
                disabled=False)

            verbose_L = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Verbose',
                disabled=False)

            warm_state_L = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                rows = 2,
                description = 'Warm_State',
                disabled = False)

            j= widgets.HBox([max_iter_L, verbose_L, warm_state_L])

            n_jobs_L = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='n_jobs',
                style = {'description_width': 'initial'},
                disabled=False)

            L1_Ratio_L = widgets.Text(
                value='None',
                placeholder='enter any integer value',
                description='L1_Ratio',
                style = {'description_width': 'initial'},
                disabled=False)

            k = widgets.HBox([n_jobs_L, L1_Ratio_L])

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            scoring_L = widgets.Dropdown(
                options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
                value = 'accuracy',
                rows = 4,
                description = 'Scoring',
                disabled = False)

            l = widgets.HBox([search_param_L, cv_L, scoring_L])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])
            
            null = widgets.HTML('<br></br>')

            button_2 = widgets.Button(description='Submit HP_Tune')
            out_res = widgets.Output()

            def on_out_res_clicked(_):
                with out_res:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
        
                    if apply_smote.value == 'y':
                        from imblearn.over_sampling import SMOTE
                        import warnings
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        from sklearn.exceptions import DataConversionWarning
                        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                        warnings.filterwarnings('ignore', 'FutureWarning') 
                        sm = SMOTE(random_state=42)
                        self.new_X, self.new_y= sm.fit_resample(self.new_X, self.new_y)
                        self.new_X = pd.DataFrame(self.new_X, columns=self.new_X.columns)
                        self.new_y= pd.DataFrame(self.new_y, columns=self.new_y.columns)
                        print('> Oversampling using SMOTE completed')
                    
                    else:
                        print('> No oversampling done')
                    
                    param_grid = {'penalty': list(penalty_L.value),
                             'dual': list(dual_L.value),
                             'tol': [float(item) for item in tol_L.value.split(',')],
                             'C' : [float(item) for item in C_L.value.split(',')],
                             'fit_intercept' : list(fit_intercept_L.value),
                             'intercept_scaling' : [float(item) for item in intercept_scaling_L.value.split(',')],
                             'class_weight' : list(class_weight_L.value),
                             'random_state' : [int(item) for item in random_state_L.value.split(',')],
                             'solver' : list(solver_L.value),
                             'max_iter' : [float(item) for item in max_iter_L.value.split(',')],
        #                      'multi_class' : list(multiclass.value),
                             'verbose' : [float(item) for item in verbose_L.value.split(',')],

        #                       'n_jobs' : [float(item) for item in n_jobs.value.split(',')]
                             }
                    
                    mlflow.sklearn.autolog()
                    warnings.filterwarnings("ignore")
                    
                    estimator = LogisticRegression()
                    
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  scoring = scoring_L.value)
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  scoring = scoring_L.value)
                    
                    with mlflow.start_run() as run:
                        warnings.filterwarnings("ignore")
                        self.classifier = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.classifier.predict(X_test)
                        acc_score= (accuracy_score(y_test,predictions))
                        roc_score= (roc_auc_score(y_test,predictions))
                        f1_macro= (f1_score(y_test, predictions, average='macro'))
                        mlflow.log_param("acc_score", acc_score)
                        mlflow.log_param("roc_score", roc_score)
                        mlflow.log_param("f1_macro", f1_macro)
                        mlflow.log_param("Best Estimator", self.classifier.best_estimator_)

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.classifier.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.classifier.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("acc_score = {:.3f} | roc_score = {:,.3f} | f1_score(macro) = {:,.3f}".format(acc_score, roc_score, f1_macro))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.classifier.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    msg = widgets.HTML('<h4>Please switch to output tab for results...</h4>')
                    msg_1 = widgets.HTML('<h4>Please run mlfow ui in command prompt to monitor HP tuning results</h4>')
                    display(msg, msg_1)
            button_2.on_click(on_out_res_clicked)
            b = widgets.VBox([button_2, out_res])

            aa = widgets.VBox([g,h,i,j,k, h5, l, m, null, b])
        # Logistic Regression Hyperpameter - Ends

        # Decision Tree Hyperparameter - Starts

            criterion_D = widgets.SelectMultiple(
                    options = ['gini', 'entropy'],
                    value = ['gini'],
                    description = 'Criterion',
                    rows = 2,
                    disabled = False)

            splitter_D = widgets.SelectMultiple(
                    options = ['best', 'random'],
                    value = ['best'],
                    rows = 2,
                    description = 'Splitter',
                    disabled = False)

            max_depth_D = widgets.Text(
                    value='5',
                    placeholder='enter any integer value',
                    description='Max_Depth',
                    disabled=False)

            min_samples_split_D = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='min_samples_split',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_samples_leaf_D = widgets.Text(
                    value='1',
                    placeholder='enter any integer value',
                    description='min_samples_leaf',
                    style = {'description_width': 'initial'},
                    disabled=False)


            min_weight_fraction_D = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='min_weight_fraction',
                    style = {'description_width': 'initial'},
                    disabled=False)

            max_features_D = widgets.SelectMultiple(
                    options = ['auto', 'sqrt', 'log2'],
                    value = ['auto'],
                    description = 'Max_Features',
                    style = {'description_width': 'initial'},
                    rows = 3,
                    disabled = False)

            random_state_D = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Random_state',
                disabled=False)

            max_leaf_nodes_D = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='Max_leaf_nodes',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_impurity_decrease_D = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='Min_impurity_decrease',
                    style = {'description_width': 'initial'},
                    disabled=False)

            class_weight_D = widgets.SelectMultiple(
                    options = ['balanced', 'None'],
                    value = ['balanced'],
                    rows = 2,
                    description = 'Class_weight',
                    style = {'description_width': 'initial'},
                    disabled = False)

            ccp_alpha_D = widgets.Text(
                    value='0.0',
                    placeholder='enter any non-negative float value',
                    description='ccp_alpha',
                    disabled=False)

            first_row = widgets.HBox([criterion_D, splitter_D, max_features_D])
            second_row = widgets.HBox([min_samples_split_D, min_weight_fraction_D, max_depth_D])
            third_row = widgets.HBox([random_state_D, max_leaf_nodes_D, min_impurity_decrease_D])
            fourth_row = widgets.HBox([ccp_alpha_D, class_weight_D, min_samples_leaf_D])

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            scoring_L = widgets.Dropdown(
                options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
                value = 'accuracy',
                rows = 4,
                description = 'Scoring',
                disabled = False)

            l = widgets.HBox([search_param_L, cv_L, scoring_L])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])
           
            
            button_3 = widgets.Button(description='Submit HP_Tune')
            out_res_DT = widgets.Output()

            def on_out_res_clicked_DT(_):
                with out_res_DT:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
        
                    if apply_smote.value == 'y':
                        from imblearn.over_sampling import SMOTE
                        import warnings
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        from sklearn.exceptions import DataConversionWarning
                        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                        warnings.filterwarnings('ignore', 'FutureWarning') 
                        sm = SMOTE(random_state=42)
                        self.new_X, self.new_y= sm.fit_resample(self.new_X, self.new_y)
                        self.new_X = pd.DataFrame(self.new_X, columns=self.new_X.columns)
                        self.new_y= pd.DataFrame(self.new_y, columns=self.new_y.columns)
                        print('> Oversampling using SMOTE completed')
                    
                    else:
                        print('> No oversampling done')
                    
                    print(criterion_D.value)
                    param_grid = {'criterion': list(criterion_D.value),
                             'splitter': list(splitter_D.value),
                             'max_depth': [int(item) for item in max_depth_D.value.split(',')],
                             'min_samples_split' : [int(item) for item in min_samples_split_D.value.split(',')],
                             'min_samples_leaf' : [int(item) for item in min_samples_leaf_D.value.split(',')],
        #                      'min_weight_fraction' : [float(item) for item in min_weight_fraction.value.split(',')],
                             'max_features' : list(max_features_D.value),
                             'random_state' : [int(item) for item in random_state_D.value.split(',')],
                             'max_leaf_nodes' : [int(item) for item in max_leaf_nodes_D.value.split(',')],
                             'min_impurity_decrease' : [float(item) for item in min_impurity_decrease_D.value.split(',')],
                             'ccp_alpha' : [float(item) for item in ccp_alpha_D.value.split(',')],
                             'class_weight' : list(class_weight_D.value)                         
                             }
                    
                    mlflow.sklearn.autolog()

                    estimator = DecisionTreeClassifier()
                    
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  scoring = scoring_L.value)
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  scoring = scoring_L.value)
                    
                    with mlflow.start_run() as run:
                        self.classifier = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.classifier.predict(X_test)
                        acc_score= (accuracy_score(y_test,predictions))
                        roc_score= (roc_auc_score(y_test,predictions))
                        f1_macro= (f1_score(y_test, predictions, average='macro'))
                        mlflow.log_param("acc_score", acc_score)
                        mlflow.log_param("roc_score", roc_score)
                        mlflow.log_param("f1_macro", f1_macro)
                        mlflow.log_param("Best Estimator", self.classifier.best_estimator_)

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.classifier.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.classifier.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("acc_score = {:.3f} | roc_score = {:,.3f} | f1_score(macro) = {:,.3f}".format(acc_score, roc_score, f1_macro))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.classifier.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    print('Please switch to output tab for results...')
            button_2.on_click(on_out_res_clicked)

            button_3.on_click(on_out_res_clicked_DT)
            b = widgets.VBox([button_3, out_res_DT])

            frame = widgets.VBox([first_row, second_row, third_row, fourth_row, h5, l, m, b])
        # Decision Tree Hyperparameter Ends

        # Random Forest Hyperparameter Starts

            n_estimators_R = widgets.Text(
                    value='100',
                    placeholder='enter any integer value',
                    description='n_estimators',
                    disabled=False)

            criterion_R = widgets.SelectMultiple(
                    options = ['gini', 'entropy'],
                    value = ['gini'],
                    rows = 2,
                    description = 'Criterion',
                    disabled = False)

            max_depth_R = widgets.Text(
                    value='5',
                    placeholder='enter any integer value',
                    description='Max_Depth',
                    disabled=False)

            min_samples_split_R = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='min_samples_split',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_samples_leaf_R = widgets.Text(
                    value='1',
                    placeholder='enter any integer value',
                    description='min_samples_leaf',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_weight_fraction_leaf_R = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='min_weight_fraction',
                    style = {'description_width': 'initial'},
                    disabled=False)

            max_features_R = widgets.SelectMultiple(
                    options = ['auto', 'sqrt', 'log2'],
                    value = ['auto'],
                    description = 'Max_Features',
                    style = {'description_width': 'initial'},
                    rows = 3,
                    disabled = False)

            random_state_R = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Random_state',
                style = {'description_width': 'initial'},
                disabled=False)

            max_leaf_nodes_R = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='Max_leaf_nodes',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_impurity_decrease_R = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='Min_impurity_decrease',
                    style = {'description_width': 'initial'},
                    disabled=False)

            bootstrap_R = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                description = 'Bootstrap',
                rows = 2,
                disabled = False)

            oob_score_R = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                description = 'oob_score',
                rows = 2,
                disabled = False)

            n_jobs_R = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='n_jobs',
                style = {'description_width': 'initial'},
                disabled=False)

            verbose_R = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Verbose',
                disabled=False)

            warm_state_R = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                description = 'Warm_State',
                style = {'description_width': 'initial'},
                disabled = False)

            class_weight_R = widgets.SelectMultiple(
                    options = ['balanced', 'balanced_subsample', 'None'],
                    value = ['balanced'],
                    description = 'Class_weight',
                    rows = 3,
                    style = {'description_width': 'initial'},
                    disabled = False)

            ccp_alpha_R = widgets.Text(
                    value='0.0',
                    placeholder='enter any non-negative float value',
                    description='ccp_alpha',
                    disabled=False)

            max_samples_R = widgets.Text(
                    value='2',
                    placeholder='enter any float value',
                    description='max_samples',
                    style = {'description_width': 'initial'},
                    disabled=False)

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            scoring_L = widgets.Dropdown(
                options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
                value = 'accuracy',
                rows = 4,
                description = 'Scoring',
                disabled = False)

            l = widgets.HBox([search_param_L, cv_L, scoring_L])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])


            first_row = widgets.HBox([n_estimators_R, criterion_R, max_depth_R])
            second_row = widgets.HBox([min_samples_split_R, min_samples_leaf_R, min_weight_fraction_leaf_R])
            third_row = widgets.HBox([max_features_R, max_leaf_nodes_R, min_impurity_decrease_R])
            fourth_row = widgets.HBox([max_samples_R, bootstrap_R, oob_score_R])
            fifth_row = widgets.HBox([n_jobs_R, random_state_R, verbose_R])
            sixth_row = widgets.HBox([warm_state_R, class_weight_R, ccp_alpha_R])

            button_4 = widgets.Button(description='Submit RF GridSearchCV')
            out_res_RF = widgets.Output()

            def on_out_res_clicked_RF(_):
                with out_res_RF:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
        
                    if apply_smote.value == 'y':
                        from imblearn.over_sampling import SMOTE
                        import warnings
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        from sklearn.exceptions import DataConversionWarning
                        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                        warnings.filterwarnings('ignore', 'FutureWarning') 
                        sm = SMOTE(random_state=42)
                        self.new_X, self.new_y= sm.fit_resample(self.new_X, self.new_y)
                        self.new_X = pd.DataFrame(self.new_X, columns=self.new_X.columns)
                        self.new_y= pd.DataFrame(self.new_y, columns=self.new_y.columns)
                        print('> Oversampling using SMOTE completed')
                    
                    else:
                        print('> No oversampling done')
                    
                    print(n_estimators_R.value)
                    param_grid = {'n_estimators' : [int(item) for item in n_estimators_R.value.split(',')],
                                  'criterion': list(criterion_R.value),
                                  'max_depth': [int(item) for item in max_depth_R.value.split(',')],
                                  'min_samples_split' : [int(item) for item in min_samples_split_R.value.split(',')],
                                  'min_samples_leaf' : [int(item) for item in min_samples_leaf_R.value.split(',')],
                                  'min_weight_fraction_leaf' : [float(item) for item in min_weight_fraction_leaf_R.value.split(',')],
                                  'max_features' : list(max_features_R.value),
                                  'random_state' : [int(item) for item in random_state_R.value.split(',')],
                                  'max_leaf_nodes' : [int(item) for item in max_leaf_nodes_R.value.split(',')],
                                  'min_impurity_decrease' : [float(item) for item in min_impurity_decrease_R.value.split(',')],
                                  'bootstrap' : list(bootstrap_R.value),
                                  'oob_score' : list(oob_score_R.value),
                                  'verbose' : [int(item) for item in verbose_R.value.split(',')],
                                  'class_weight' : list(class_weight_R.value),
                                  'ccp_alpha' : [float(item) for item in ccp_alpha_R.value.split(',')],
                                  'max_samples' : [int(item) for item in max_samples_R.value.split(',')]                         
                             }

                    mlflow.sklearn.autolog()

                    estimator = RandomForestClassifier()
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  scoring = scoring_L.value)
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  scoring = scoring_L.value)
                    
                    with mlflow.start_run() as run:
                        self.classifier = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.classifier.predict(X_test)
                        acc_score= (accuracy_score(y_test,predictions))
                        roc_score= (roc_auc_score(y_test,predictions))
                        f1_macro= (f1_score(y_test, predictions, average='macro'))
                        mlflow.log_param("acc_score", acc_score)
                        mlflow.log_param("roc_score", roc_score)
                        mlflow.log_param("f1_macro", f1_macro)
                        mlflow.log_param("Best Estimator", self.classifier.best_estimator_)
                        print('Running Grid Search...')

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.classifier.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.classifier.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("acc_score = {:.3f} | roc_score = {:,.3f} | f1_score(macro) = {:,.3f}".format(acc_score, roc_score, f1_macro))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.classifier.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    print('Grid Search completed, Please switch to output tab for results...')
            # button_2.on_click(on_out_res_clicked)

            button_4.on_click(on_out_res_clicked_RF)
            b = widgets.VBox([button_4, out_res_RF])

            frame_RF = widgets.VBox([first_row, second_row, third_row, fourth_row, fifth_row, sixth_row, h5, l, m, b])
        # Random Forest Hyperparameter ends


        # KNN Classifier Hyperparameter Starts
            n_neighbors_k = widgets.Text(
                    value='5',
                    placeholder='enter any integer value',
                    description='n_neighbors',
                    disabled=False)
            
            weights_k = widgets.SelectMultiple(
                options = ['uniform', 'distance'],
                value = ['uniform'],
                rows = 2,
                description = 'Weights',
                disabled = False)

            algorithm_k = widgets.SelectMultiple(
                options = ['auto', 'ball_tree', 'kd_tree', 'brute'],
                value = ['auto'],
                rows = 4,
                description = 'Algorithm',
                disabled = False)

            leaf_size_k = widgets.Text(
                    value='30',
                    placeholder='enter any integer value',
                    description='Leaf_Size',
                    disabled=False)
            
            p_k = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='p (Power param)',
                    disabled=False)
            
            metric_k = widgets.SelectMultiple(
                options = ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                value = ['minkowski'],
                rows = 4,
                description = 'Metric',
                disabled = False)

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            scoring_L = widgets.Dropdown(
                options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
                value = 'accuracy',
                rows = 4,
                description = 'Scoring',
                disabled = False)

            l = widgets.HBox([search_param_L, cv_L, scoring_L])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])


            first_row = widgets.HBox([n_neighbors_k, weights_k, algorithm_k])
            second_row = widgets.HBox([leaf_size_k, p_k, metric_k])

            button_5 = widgets.Button(description='Submit RF GridSearchCV')
            out_res_K = widgets.Output()

            def on_out_res_clicked_K(_):
                with out_res_K:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
        
                    if apply_smote.value == 'y':
                        from imblearn.over_sampling import SMOTE
                        import warnings
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        from sklearn.exceptions import DataConversionWarning
                        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                        warnings.filterwarnings('ignore', 'FutureWarning') 
                        sm = SMOTE(random_state=42)
                        self.new_X, self.new_y= sm.fit_resample(self.new_X, self.new_y)
                        self.new_X = pd.DataFrame(self.new_X, columns=self.new_X.columns)
                        self.new_y= pd.DataFrame(self.new_y, columns=self.new_y.columns)
                        print('> Oversampling using SMOTE completed')
                    
                    else:
                        print('> No oversampling done')
                    
                    print(n_neighbors_k.value)
                    param_grid = {'n_neighbors' : [int(item) for item in n_neighbors_k.value.split(',')],
                                  'weights': list(weights_k.value),
                                  'algorithm': list(algorithm_k.value),
                                  'leaf_size' : [int(item) for item in leaf_size_k.value.split(',')],
                                  'p' : [int(item) for item in p_k.value.split(',')],
                                  'metric' : list(metric_k.value),
                             }

                    mlflow.sklearn.autolog()

                    estimator = KNeighborsClassifier()
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  scoring = scoring_L.value)
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  scoring = scoring_L.value)
                    
                    with mlflow.start_run() as run:
                        self.classifier = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.classifier.predict(X_test)
                        acc_score= (accuracy_score(y_test,predictions))
                        roc_score= (roc_auc_score(y_test,predictions))
                        f1_macro= (f1_score(y_test, predictions, average='macro'))
                        mlflow.log_param("acc_score", acc_score)
                        mlflow.log_param("roc_score", roc_score)
                        mlflow.log_param("f1_macro", f1_macro)
                        mlflow.log_param("Best Estimator", self.classifier.best_estimator_)
                        print('Running Grid Search...')

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.classifier.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.classifier.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("acc_score = {:.3f} | roc_score = {:,.3f} | f1_score(macro) = {:,.3f}".format(acc_score, roc_score, f1_macro))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.classifier.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    print('Grid Search completed, Please switch to output tab for results...')
            # button_2.on_click(on_out_res_clicked)

            button_5.on_click(on_out_res_clicked_K)
            b = widgets.VBox([button_5, out_res_K])

            frame_K = widgets.VBox([first_row, second_row, h5, l, m, b])
            
    def HP_Tune_Regression(self):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.linear_model import LinearRegression
        import mlflow
        from mlflow import log_metric, log_param, log_artifacts
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor 
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from math import sqrt
        import ipywidgets as widgets
        from ipywidgets import HBox, VBox, Button, Label
        from ipywidgets import  interact_manual, interactive, interact
        import logging
        from IPython.display import display, Markdown, clear_output
        import warnings
        warnings.filterwarnings('ignore')

        out1 = widgets.Output()
        out2 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2])
        tab.set_title(0, 'Input')
        tab.set_title(1, 'Output')

        display(tab)

        with out1:
            header_1 = widgets.HTML(value="<h2>Pywedge HP_Tune</h2>")
            header = widgets.HTML(value="<h3>Base Estimator</h3>")
            display(header_1, header)

            import pandas as pd
            cat_info = widgets.Dropdown(
                options = [('cat_codes', '1'), ('get_dummies', '2')],
                value = '1',
                description = 'Select categorical conversion',
                style = {'description_width': 'initial'},     
                disabled=False)

            std_scr = widgets.Dropdown(
                options = [('StandardScalar', '1'), ('RobustScalar', '2'), ('MinMaxScalar', '3'), ('No Standardization', 'n')],
                value = 'n',
                description = 'Select Standardization methods',
                style = {'description_width': 'initial'},     
                disabled=False)

            pp_class = widgets.HBox([cat_info, std_scr])

            header_2 = widgets.HTML(value="<h3>Pre_processing </h3>")

            base_estimator = widgets.Dropdown(
            options=['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'KNN Regressor'],
            value='Linear Regression',
            description='Choose Base Estimator: ',
            style = {'description_width': 'initial'},     
            disabled=False)

            display(base_estimator)


            button = widgets.Button(description='Select Base Estimator')
            out = widgets.Output()

            def on_button_clicked(_):
                with out:
                    clear_output()
                    selection = base_estimator.value
                    if selection == 'Linear Regression':
                        h1 = widgets.HTML('<h3>Select Linear Regression Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(aa)
                    elif selection =='Decision Tree Regressor':
                        h1 = widgets.HTML('<h3>Select Decision Tree Regressor Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(frame)
                    elif selection == 'Random Forest Regressor':
                        h1 = widgets.HTML('<h3>Select Random Forest Regressor Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(frame_RF)
                    elif selection == 'KNN Regressor':
                        h1 = widgets.HTML('<h3>Select KNN Regressor Hyperparameters</h3>')
                        display(header_2, pp_class)
                        display(h1)
                        display(frame_K)

            button.on_click(on_button_clicked)

            a = widgets.VBox([button, out])
            display(a)

        # Linear Regression Hyperparameters _Start

            fit_intercept_L = widgets.SelectMultiple(
                options = [True, False],
                value = [True],
                rows = 2,       
                description = 'Fit_Intercept',
                disabled = False)

            normalize_L = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                rows = 2,       
                description = 'Normalize',
                disabled = False)

            copy_X_L = widgets.SelectMultiple(
                options = [True, False],
                value = [True],
                rows = 2,       
                description = 'Copy_X',
                disabled = False)

            g = widgets.HBox([fit_intercept_L, normalize_L, copy_X_L])

            positive_L = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                rows = 2,       
                description = 'Positive',
                disabled = False)

            h = widgets.HBox([positive_L])

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            scoring_L = widgets.Dropdown(
                options = ['explained_variance', 'neg_mean_squared_error', 'r2'],
                value = 'explained_variance',
                rows = 3,
                description = 'Scoring',
                disabled = False)

            l = widgets.HBox([search_param_L, cv_L, scoring_L])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])
            
            null = widgets.HTML('<br></br>')

            button_2 = widgets.Button(description='Submit HP_Tune')
            out_res = widgets.Output()

            def on_out_res_clicked(_):
                with out_res:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
                    
                    param_grid = {'fit_intercept': list(fit_intercept_L.value),
                             'normalize': list(normalize_L.value),
                             'copy_X': list(copy_X_L.value)
                             #'positive' : list(positive_L.value)
                             }
                    
                    mlflow.sklearn.autolog()
                    warnings.filterwarnings("ignore")
                    
                    estimator = LinearRegression()
                    
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  scoring = scoring_L.value)
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  scoring = scoring_L.value)
                    
                    with mlflow.start_run() as run:
                        warnings.filterwarnings("ignore")
                        self.regressor = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.regressor.predict(X_test)
                        exp_var = explained_variance_score(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        rmse = sqrt(mean_absolute_error(y_test, predictions))
                        r2 = r2_score(y_test, predictions)
                        mlflow.log_param("Exp_Var", exp_var)
                        mlflow.log_param("MAE", mae)
                        mlflow.log_param("RMSE", rmse)
                        mlflow.log_param('R2', r2)
                        mlflow.log_param("Best Estimator", self.regressor.best_estimator_)

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.regressor.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.regressor.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} ".format(exp_var, mae, rmse, r2,))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.regressor.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    msg = widgets.HTML('<h4>Please switch to output tab for results...</h4>')
                    msg_1 = widgets.HTML('<h4>Please run mlfow ui in command prompt to monitor HP tuning results</h4>')
                    display(msg, msg_1)
            button_2.on_click(on_out_res_clicked)
            b = widgets.VBox([button_2, out_res])

            aa = widgets.VBox([g,h,h5, l, m, null, b])
        # Linear Regression Hyperpameter - Ends

        # Decision Tree Regressor Hyperparameter - Starts

            criterion_D = widgets.SelectMultiple(
                    options = ['mse', 'friedman_mse', 'mae', 'poisson'],
                    value = ['mse'],
                    description = 'Criterion',
                    rows = 4,
                    disabled = False)

            splitter_D = widgets.SelectMultiple(
                    options = ['best', 'random'],
                    value = ['best'],
                    rows = 2,
                    description = 'Splitter',
                    disabled = False)

            max_depth_D = widgets.Text(
                    value='5',
                    placeholder='enter any integer value',
                    description='Max_Depth',
                    disabled=False)

            min_samples_split_D = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='min_samples_split',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_samples_leaf_D = widgets.Text(
                    value='1',
                    placeholder='enter any integer value',
                    description='min_samples_leaf',
                    style = {'description_width': 'initial'},
                    disabled=False)


            min_weight_fraction_D = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='min_weight_fraction',
                    style = {'description_width': 'initial'},
                    disabled=False)

            max_features_D = widgets.SelectMultiple(
                    options = ['auto', 'sqrt', 'log2'],
                    value = ['auto'],
                    description = 'Max_Features',
                    style = {'description_width': 'initial'},
                    rows = 3,
                    disabled = False)

            random_state_D = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Random_state',
                disabled=False)

            max_leaf_nodes_D = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='Max_leaf_nodes',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_impurity_decrease_D = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='Min_impurity_decrease',
                    style = {'description_width': 'initial'},
                    disabled=False)

            ccp_alpha_D = widgets.Text(
                    value='0.0',
                    placeholder='enter any non-negative float value',
                    description='ccp_alpha',
                    disabled=False)

            first_row = widgets.HBox([criterion_D, splitter_D, max_features_D])
            second_row = widgets.HBox([min_samples_split_D, min_weight_fraction_D, max_depth_D])
            third_row = widgets.HBox([random_state_D, max_leaf_nodes_D, min_impurity_decrease_D])
            fourth_row = widgets.HBox([ccp_alpha_D, min_samples_leaf_D])

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_jobs_L = widgets.Text(
                value='1',
                placeholder='enter any integer value',
                description='n_jobs',
                style = {'description_width': 'initial'},
                disabled=False)

            # scoring_L = widgets.Dropdown(
            #     options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
            #     value = 'accuracy',
            #     rows = 4,
            #     description = 'Scoring',
            #     disabled = False)

            l = widgets.HBox([search_param_L, cv_L, n_jobs_L])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])
           
            
            button_3 = widgets.Button(description='Submit HP_Tune')
            out_res_DT = widgets.Output()

            def on_out_res_clicked_DT(_):
                with out_res_DT:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
                    
                    print(criterion_D.value)
                    param_grid = {'criterion': list(criterion_D.value),
                             'splitter': list(splitter_D.value),
                             'max_depth': [int(item) for item in max_depth_D.value.split(',')],
                             'min_samples_split' : [int(item) for item in min_samples_split_D.value.split(',')],
                             'min_samples_leaf' : [int(item) for item in min_samples_leaf_D.value.split(',')],
        #                      'min_weight_fraction' : [float(item) for item in min_weight_fraction.value.split(',')],
                             'max_features' : list(max_features_D.value),
                             'random_state' : [int(item) for item in random_state_D.value.split(',')],
                             'max_leaf_nodes' : [int(item) for item in max_leaf_nodes_D.value.split(',')],
                             'min_impurity_decrease' : [float(item) for item in min_impurity_decrease_D.value.split(',')],
                             'ccp_alpha' : [float(item) for item in ccp_alpha_D.value.split(',')]
                             }
                    
                    mlflow.sklearn.autolog()

                    estimator = DecisionTreeRegressor()
                    
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  n_jobs = int(cv_L.value))
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  n_jobs = int(cv_L.value))
                    
                    with mlflow.start_run() as run:
                        warnings.filterwarnings("ignore")
                        self.regressor = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.regressor.predict(X_test)
                        exp_var = explained_variance_score(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        rmse = sqrt(mean_absolute_error(y_test, predictions))
                        r2 = r2_score(y_test, predictions)
                        mlflow.log_param("Exp_Var", exp_var)
                        mlflow.log_param("MAE", mae)
                        mlflow.log_param("RMSE", rmse)
                        mlflow.log_param('R2', r2)
                        mlflow.log_param("Best Estimator", self.regressor.best_estimator_)

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.regressor.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.regressor.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} ".format(exp_var, mae, rmse, r2,))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.regressor.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    print('Please switch to output tab for results...')
            button_2.on_click(on_out_res_clicked)

            button_3.on_click(on_out_res_clicked_DT)
            b = widgets.VBox([button_3, out_res_DT])

            frame = widgets.VBox([first_row, second_row, third_row, fourth_row, h5, l, m, b])
        # Decision Tree Hyperparameter Ends

        # Random Forest Regressor Hyperparameter Starts

            n_estimators_R = widgets.Text(
                    value='100',
                    placeholder='enter any integer value',
                    description='n_estimators',
                    disabled=False)

            criterion_R = widgets.SelectMultiple(
                    options = ['mse', 'mae'],
                    value = ['mse'],
                    rows = 2,
                    description = 'Criterion',
                    disabled = False)

            max_depth_R = widgets.Text(
                    value='5',
                    placeholder='enter any integer value',
                    description='Max_Depth',
                    disabled=False)

            min_samples_split_R = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='min_samples_split',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_samples_leaf_R = widgets.Text(
                    value='1',
                    placeholder='enter any integer value',
                    description='min_samples_leaf',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_weight_fraction_leaf_R = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='min_weight_fraction',
                    style = {'description_width': 'initial'},
                    disabled=False)

            max_features_R = widgets.SelectMultiple(
                    options = ['auto', 'sqrt', 'log2'],
                    value = ['auto'],
                    description = 'Max_Features',
                    style = {'description_width': 'initial'},
                    rows = 3,
                    disabled = False)

            random_state_R = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Random_state',
                style = {'description_width': 'initial'},
                disabled=False)

            max_leaf_nodes_R = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='Max_leaf_nodes',
                    style = {'description_width': 'initial'},
                    disabled=False)

            min_impurity_decrease_R = widgets.Text(
                    value='0.0',
                    placeholder='enter any float value',
                    description='Min_impurity_decrease',
                    style = {'description_width': 'initial'},
                    disabled=False)

            bootstrap_R = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                description = 'Bootstrap',
                rows = 2,
                disabled = False)

            oob_score_R = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                description = 'oob_score',
                rows = 2,
                disabled = False)

            verbose_R = widgets.Text(
                value='0',
                placeholder='enter any integer value',
                description='Verbose',
                disabled=False)

            warm_state_R = widgets.SelectMultiple(
                options = [True, False],
                value = [False],
                description = 'Warm_State',
                rows = 2,
                style = {'description_width': 'initial'},
                disabled = False)

            class_weight_R = widgets.SelectMultiple(
                    options = ['balanced', 'balanced_subsample', 'None'],
                    value = ['balanced'],
                    description = 'Class_weight',
                    rows = 3,
                    style = {'description_width': 'initial'},
                    disabled = False)

            ccp_alpha_R = widgets.Text(
                    value='0.0',
                    placeholder='enter any non-negative float value',
                    description='ccp_alpha',
                    disabled=False)

            max_samples_R = widgets.Text(
                    value='2',
                    placeholder='enter any float value',
                    description='max_samples',
                    style = {'description_width': 'initial'},
                    disabled=False)

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_jobs_R = widgets.Text(
                value='1',
                placeholder='enter any integer value',
                description='n_jobs',
                style = {'description_width': 'initial'},
                disabled=False)

            # scoring_L = widgets.Dropdown(
            #     options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
            #     value = 'accuracy',
            #     rows = 4,
            #     description = 'Scoring',
            #     disabled = False)

            l = widgets.HBox([search_param_L, cv_L, n_jobs_R])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])


            first_row = widgets.HBox([n_estimators_R, criterion_R, max_depth_R])
            second_row = widgets.HBox([min_samples_split_R, min_samples_leaf_R, min_weight_fraction_leaf_R])
            third_row = widgets.HBox([max_features_R, max_leaf_nodes_R, min_impurity_decrease_R])
            fourth_row = widgets.HBox([max_samples_R, bootstrap_R, oob_score_R])
            fifth_row = widgets.HBox([random_state_R, verbose_R])
            sixth_row = widgets.HBox([warm_state_R, ccp_alpha_R])

            button_4 = widgets.Button(description='Submit RF GridSearchCV')
            out_res_RF = widgets.Output()

            def on_out_res_clicked_RF(_):
                with out_res_RF:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
                    
                    print(n_estimators_R.value)
                    param_grid = {'n_estimators' : [int(item) for item in n_estimators_R.value.split(',')],
                                  'criterion': list(criterion_R.value),
                                  'max_depth': [int(item) for item in max_depth_R.value.split(',')],
                                  'min_samples_split' : [int(item) for item in min_samples_split_R.value.split(',')],
                                  'min_samples_leaf' : [int(item) for item in min_samples_leaf_R.value.split(',')],
                                  'min_weight_fraction_leaf' : [float(item) for item in min_weight_fraction_leaf_R.value.split(',')],
                                  'max_features' : list(max_features_R.value),
                                  'random_state' : [int(item) for item in random_state_R.value.split(',')],
                                  'max_leaf_nodes' : [int(item) for item in max_leaf_nodes_R.value.split(',')],
                                  'min_impurity_decrease' : [float(item) for item in min_impurity_decrease_R.value.split(',')],
                                  'bootstrap' : list(bootstrap_R.value),
                                  'oob_score' : list(oob_score_R.value),
                                  'verbose' : [int(item) for item in verbose_R.value.split(',')],
                                  'ccp_alpha' : [float(item) for item in ccp_alpha_R.value.split(',')],
                                  'max_samples' : [int(item) for item in max_samples_R.value.split(',')]                         
                             }

                    mlflow.sklearn.autolog()

                    estimator = RandomForestRegressor()
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  n_jobs = int(n_jobs_R.value))
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  n_jobs = int(n_jobs_R.value))
                    
                    with mlflow.start_run() as run:
                        warnings.filterwarnings("ignore")
                        self.regressor = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.regressor.predict(X_test)
                        exp_var = explained_variance_score(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        rmse = sqrt(mean_absolute_error(y_test, predictions))
                        r2 = r2_score(y_test, predictions)
                        mlflow.log_param("Exp_Var", exp_var)
                        mlflow.log_param("MAE", mae)
                        mlflow.log_param("RMSE", rmse)
                        mlflow.log_param('R2', r2)
                        mlflow.log_param("Best Estimator", self.regressor.best_estimator_)

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.regressor.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.regressor.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} ".format(exp_var, mae, rmse, r2,))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.regressor.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    print('Grid Search completed, Please switch to output tab for results...')
            # button_2.on_click(on_out_res_clicked)

            button_4.on_click(on_out_res_clicked_RF)
            b = widgets.VBox([button_4, out_res_RF])

            frame_RF = widgets.VBox([first_row, second_row, third_row, fourth_row, fifth_row, sixth_row, h5, l, m, b])
        # Random Forest Regressor Hyperparameter ends


        # KNN Regressor Hyperparameter Starts
            n_neighbors_k = widgets.Text(
                    value='5',
                    placeholder='enter any integer value',
                    description='n_neighbors',
                    disabled=False)
            
            weights_k = widgets.SelectMultiple(
                options = ['uniform', 'distance'],
                value = ['uniform'],
                rows = 2,
                description = 'Weights',
                disabled = False)

            algorithm_k = widgets.SelectMultiple(
                options = ['auto', 'ball_tree', 'kd_tree', 'brute'],
                value = ['auto'],
                rows = 4,
                description = 'Algorithm',
                disabled = False)

            leaf_size_k = widgets.Text(
                    value='30',
                    placeholder='enter any integer value',
                    description='Leaf_Size',
                    disabled=False)
            
            p_k = widgets.Text(
                    value='2',
                    placeholder='enter any integer value',
                    description='p (Power param)',
                    disabled=False)
            
            metric_k = widgets.SelectMultiple(
                options = ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                value = ['minkowski'],
                rows = 4,
                description = 'Metric',
                disabled = False)

            h5 = widgets.HTML('<h4>Select Grid/Random search Hyperparameters</h4>')
            
            search_param_L = widgets.Dropdown(
                options=['GridSearch CV', 'Random Search CV'],
                value='GridSearch CV',
                description='Choose Search Option: ',
                style = {'description_width': 'initial'},     
                disabled=False)

            cv_L = widgets.Text(
                value='5',
                placeholder='enter any integer value',
                description='CV',
                style = {'description_width': 'initial'},
                disabled=False)

            n_jobs_R = widgets.Text(
                value='1',
                placeholder='enter any integer value',
                description='n_jobs',
                style = {'description_width': 'initial'},
                disabled=False)

            
            # scoring_L = widgets.Dropdown(
            #     options = ['accuracy', 'f1', 'roc_auc', 'balanced_accuracy'],
            #     value = 'accuracy',
            #     rows = 4,
            #     description = 'Scoring',
            #     disabled = False)

            l = widgets.HBox([search_param_L, cv_L, n_jobs_R])

            n_iter_L = widgets.Text(
                value='10',
                placeholder='enter any integer value',
                description='n_iter',
                style = {'description_width': 'initial'},
                disabled=False)
            
            n_iter_text = widgets.HTML(value='<p><em>For Random Search</em></p>')

            m = widgets.HBox([n_iter_L, n_iter_text])


            first_row = widgets.HBox([n_neighbors_k, weights_k, algorithm_k])
            second_row = widgets.HBox([leaf_size_k, p_k, metric_k])

            button_5 = widgets.Button(description='Submit RF GridSearchCV')
            out_res_K = widgets.Output()

            def on_out_res_clicked_K(_):
                with out_res_K:
                    clear_output()
                    import pandas as pd
                    self.new_X = self.X.copy(deep=True)
                    self.new_y = self.y
                    self.new_test = self.test.copy(deep=True)
                    categorical_cols = self.new_X.select_dtypes('object').columns.to_list()
                    for col in categorical_cols:
                        self.new_X[col].fillna(self.new_X[col].mode()[0], inplace=True)
                    numeric_cols = self.new_X.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_X[col].fillna(self.new_X[col].mean(), inplace=True)
    
                    test_categorical_cols = self.new_test.select_dtypes('object').columns.to_list()
                    for col in test_categorical_cols:
                        self.new_test[col].fillna(self.new_test[col].mode()[0], inplace=True)
                    numeric_cols = self.new_test.select_dtypes(['float64', 'int64']).columns.to_list()
                    for col in numeric_cols:
                        self.new_test[col].fillna(self.new_test[col].mean(), inplace=True)
                    
                    if cat_info.value == '1':
                        for col in categorical_cols:
                            self.new_X[col] = self.new_X[col].astype('category')
                            self.new_X[col] = self.new_X[col].cat.codes
                            self.new_test[col] = self.new_test[col].astype('category')
                            self.new_test[col] = self.new_test[col].cat.codes
                        print('> Categorical columns converted using Catcodes')
                    if cat_info.value == '2':        
                        self.new_X = pd.get_dummies(self.new_X,drop_first=True)
                        self.new_test = pd.get_dummies(self.new_test,drop_first=True)
                        print('> Categorical columns converted using Get_Dummies')
                    self.new_y = pd.DataFrame(self.train[[self.y]])
                    self.new_y = pd.get_dummies(self.new_y,drop_first=True)
                    
                    if std_scr.value == '1':
                        from sklearn.preprocessing import StandardScaler
                        scalar = StandardScaler()
                        self.new_X = pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test = pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Standard Scalar completed')
                    elif std_scr.value == '2':
                        from sklearn.preprocessing import RobustScaler
                        scalar = RobustScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Roubust Scalar completed')
                        
                    elif std_scr.value == '3':
                        from sklearn.preprocessing import MinMaxScaler
                        scalar = MinMaxScaler()
                        self.new_X= pd.DataFrame(scalar.fit_transform(self.new_X), columns=self.new_X.columns, index=self.new_X.index)
                        self.new_test= pd.DataFrame(scalar.fit_transform(self.new_test), columns=self.new_test.columns, index=self.new_test.index)
                        print('> standardization using Minmax Scalar completed')
                    
                    elif std_scr.value == 'n':
                        print('> No standardization done')
                    
                    print(n_neighbors_k.value)
                    param_grid = {'n_neighbors' : [int(item) for item in n_neighbors_k.value.split(',')],
                                  'weights': list(weights_k.value),
                                  'algorithm': list(algorithm_k.value),
                                  'leaf_size' : [int(item) for item in leaf_size_k.value.split(',')],
                                  'p' : [int(item) for item in p_k.value.split(',')],
                                  'metric' : list(metric_k.value),
                             }

                    mlflow.sklearn.autolog()

                    estimator = KNeighborsRegressor()
                    if search_param_L.value == 'GridSearch CV':
                        grid_lr = GridSearchCV(estimator=estimator,
                              param_grid = param_grid,
                                  cv = int(cv_L.value),
                                  n_jobs = int(n_jobs_R.value))
                        
                    if search_param_L.value == 'Random Search CV':
                        grid_lr = RandomizedSearchCV(estimator=estimator,
                              param_distributions = param_grid,
                                  cv = int(cv_L.value),
                                n_iter = int(n_iter_L.value),
                                  n_jobs = int(n_jobs_R.value))
                    
                    with mlflow.start_run() as run:
                        warnings.filterwarnings("ignore")
                        self.regressor = grid_lr.fit(self.new_X.values, self.new_y.values)
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score    
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.new_X.values, self.new_y.values, test_size=0.2, random_state=1)
                        predictions = self.regressor.predict(X_test)
                        exp_var = explained_variance_score(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        rmse = sqrt(mean_absolute_error(y_test, predictions))
                        r2 = r2_score(y_test, predictions)
                        mlflow.log_param("Exp_Var", exp_var)
                        mlflow.log_param("MAE", mae)
                        mlflow.log_param("RMSE", rmse)
                        mlflow.log_param('R2', r2)
                        mlflow.log_param("Best Estimator", self.regressor.best_estimator_)

                    with out2:
                        clear_output()
                        GP = widgets.HTML(value='<h3><em>Get_Params</em></h3>')
                        display(GP)
                        print(self.regressor.get_params)
                        BE = widgets.HTML(value='<h3><em>Best Estimator</em></h3>')
                        display(BE)
                        print(self.regressor.best_estimator_)
                        BS = widgets.HTML(value='<h3><em>Metrics on Train data</em></h3>')
                        display(BS)
                        print("exp_var = {:.3f} | mae = {:,.3f} | rmse = {:,.3f} | r2 = {:,.3f} ".format(exp_var, mae, rmse, r2,))
                        Pred = widgets.HTML(value='<h3><em>Predictions on stand_out test data</em></h3>')
                        display(Pred)
                        self.predict_HP = self.regressor.predict(self.new_test)
                        print('Prediction completed. \nUse dot operator in below code cell to access predict, for eg., pph.predict_HP, where pph is pywedge_HP class object')
                    print('Grid Search completed, Please switch to output tab for results...')
            # button_2.on_click(on_out_res_clicked)

            button_5.on_click(on_out_res_clicked_K)
            b = widgets.VBox([button_5, out_res_K])

            frame_K = widgets.VBox([first_row, second_row, h5, l, m, b])

