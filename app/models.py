import os
import sys
sys.path.insert(0, '../src/')
import datetime
import time
from pathlib import Path
from prettytable import PrettyTable

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import auto_co2 as co2

########## CLASSIFICATION MODELS ##########


class MyDecisionTreeClassifier(DecisionTreeClassifier):
    DEFAULT_HYPERPARAMS = { # selection of hyperparameters for user prompt
        'criterion': 'gini',
        'max_depth': 20,
        'min_samples_leaf': 1,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'class_weight': None
    }    
    def __init__(self, 
                 criterion='gini', 
                 splitter='best', 
                 max_depth=None, 
                 min_samples_split=2, 
                 min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, 
                 max_features=None, 
                 random_state=None, 
                 max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, 
                 class_weight=None, 
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion, splitter=splitter, max_depth=max_depth, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
            random_state=random_state, max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, 
            ccp_alpha=ccp_alpha
            )
            
class MyRandomForestClassifier(RandomForestClassifier):
    DEFAULT_HYPERPARAMS = {
        'criterion': 'gini',
        'max_depth': None,
        'n_estimators': 100,
        'max_features': 'sqrt', #int, float, None=n_features, log2, sqrt = log2(n_features) etc
        'max_leaf_nodes': None,
        'bootstrap': False,
    } 
    def __init__(self, 
                 n_estimators=100, 
                 criterion='gini', 
                 max_depth=None, 
                 min_samples_split=2, 
                 min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, 
                 max_features='auto', 
                 max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, 
                 bootstrap=True, 
                 oob_score=False, 
                 n_jobs=-1, 
                 random_state=None, 
                 verbose=2, 
                 warm_start=False, 
                 class_weight=None, 
                 ccp_alpha=0.0, 
                 max_samples=None):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, 
            bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, 
            random_state=random_state, verbose=verbose, warm_start=warm_start, 
            class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples
            )

class MyXGBoostClassifier(xgb.XGBClassifier):
    DEFAULT_HYPERPARAMS = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 100,
        'gamma': 0,
        'lambda': 1,
        'objective': 'multi:softmax'
        }
    def __init__(self, 
                 max_depth=3, 
                 learning_rate=0.1, 
                 n_estimators=100, 
                 verbosity=1, 
                 objective='binary:logistic', 
                 booster='gbtree', 
                 n_jobs=-1, 
                 gamma=0, 
                 min_child_weight=1, 
                 max_delta_step=0, 
                 subsample=1, 
                 colsample_bytree=1, 
                 colsample_bylevel=1, 
                 colsample_bynode=1, 
                 reg_alpha=0, 
                 reg_lambda=1, 
                 scale_pos_weight=1, 
                 base_score=0.5, 
                 random_state=0, 
                 num_parallel_tree=1, 
                 importance_type='gain', 
                 eval_metric='logloss'):
        super().__init__(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
            verbosity=verbosity, objective=objective, booster=booster, 
            n_jobs=n_jobs, gamma=gamma, min_child_weight=min_child_weight, 
            max_delta_step=max_delta_step, subsample=subsample, 
            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, 
            colsample_bynode=colsample_bynode, reg_alpha=reg_alpha, 
            reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, 
            base_score=base_score, random_state=random_state, 
            num_parallel_tree=num_parallel_tree, importance_type=importance_type, 
            eval_metric=eval_metric
        )

                 
class MyLinearRegression(LinearRegression):
    DEFAULT_HYPERPARAMS = {
        'fit_intercept': True,
        'positive': False,
        'copy_X': True,
        'n_jobs': -1}
    def __init__(self, 
                 fit_intercept=True, 
                 positive=False, 
                 copy_X=True, 
                 n_jobs=-1):
        super().__init__(
            fit_intercept=fit_intercept, positive=positive, 
            copy_X=copy_X, n_jobs=n_jobs
        )
        
class MyElasticNetCV(ElasticNetCV):
    DEFAULT_HYPERPARAMS = {
        'l1_ratio': 0.5,
        'cv': 3,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 0.0001,
        'eps': 0.001,
        }
    def __init__(self, 
                 n_alphas=100, 
                 l1_ratio=0.5, 
                 eps=1e-3,
                 fit_intercept=True,  
                 precompute=False, 
                 max_iter=1000, 
                 copy_X=True, 
                 tol=0.0001, 
                 positive=False, 
                 random_state=None, 
                 selection='cyclic',
                 cv=3,
                 n_jobs=-1,
                 verbose=1):
        super().__init__(
            n_alphas=n_alphas, eps=eps, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
            precompute=precompute, max_iter=max_iter, copy_X=copy_X, tol=tol, n_jobs=n_jobs,
            positive=positive, random_state=random_state, selection=selection,
            verbose=verbose, cv=cv
            )
        
        

class MyXGBoostRegressor(xgb.XGBRegressor):
    DEFAULT_HYPERPARAMS = {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        }

    def __init__(self, 
                 max_depth=3, 
                 learning_rate=0.1, 
                 n_estimators=100, 
                 verbosity=2, 
                 objective='reg:squarederror', 
                 booster='gbtree', 
                 n_jobs=-1, 
                 gamma=0, 
                 min_child_weight=1, 
                 max_delta_step=0, 
                 subsample=1, 
                 colsample_bytree=1, 
                 colsample_bylevel=1, 
                 colsample_bynode=1, 
                 reg_alpha=0, 
                 reg_lambda=1, 
                 scale_pos_weight=1, 
                 base_score=0.5, 
                 random_state=0, 
                 num_parallel_tree=1, 
                 importance_type='gain', 
                 eval_metric='rmse'):
        super().__init__(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
            verbosity=verbosity, objective=objective, booster=booster, 
            n_jobs=n_jobs, gamma=gamma, min_child_weight=min_child_weight, 
            max_delta_step=max_delta_step, subsample=subsample, 
            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, 
            colsample_bynode=colsample_bynode, reg_alpha=reg_alpha, 
            reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, 
            base_score=base_score, random_state=random_state, 
            num_parallel_tree=num_parallel_tree, importance_type=importance_type, 
            eval_metric=eval_metric
        )
        
                 
class MLToolBox:  

    @staticmethod
    def load_data(filepath):
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.pkl'):
            df = pd.read_pickle(filepath)
        else:
            raise ValueError("Unsupported file type. Please provide a .csv or .pkl file.")
        print(df.info())
        return df
        

    @staticmethod
    def separate_target(df, classification):
        if classification:
            X = df.drop(columns=['Co2Grade'])
            y = df['Co2Grade']
        else:
            X = df.drop(columns=['Co2EmissionsWltp'])
            y = df['Co2EmissionsWltp']
        print(f"Target info: shape = {y.shape}, values = {y.value_counts()}")
        return X, y
        
    
    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def standard_scale(X, feature_df=True):
        scaler = StandardScaler()
        if feature_df:
            return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            return scaler.fit_transform(X)
    
    @staticmethod
    def minmax_scale(X, feature_df=True):
        scaler = MinMaxScaler()
        if feature_df:
            return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            return scaler.fit_transform(X)
        
    
    @staticmethod
    def prepare(dataset_path, test_size=0.2, random_state=None, feature_scaling=0, classification=False):
        # combines toolbox methods to prepare the dataset for training
        df = MLToolBox.load_data(dataset_path)
        X, y  = MLToolBox.separate_target(df, classification=classification)
        print(f"y shape: {y.shape}, type: {type(y)}")  # add this line
        data_shape = X.shape
        print(f"Loaded {dataset_path}")
        
        if feature_scaling == 1: # minmax normalisation
            X = MLToolBox.minmax_scale(X)
            print("Data has been normalized")    
        elif feature_scaling == 2: #standardisation
            X = MLToolBox.standard_scale(X)
            print("Data has been standardized")
        else:
            print("No feature scaling applied")
            
        X_train, X_test, y_train, y_test = MLToolBox.split_data(X, y, test_size=test_size, random_state=random_state)
        print(f"Data has been split into train and test sets with a test size of {test_size}, random state: {random_state}")
        print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")  # add this line



        return X_train, X_test, y_train, y_test, data_shape
    
    
    @staticmethod
    def compare_results(y_true, y_pred):
        df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        df.index.name = 'Row Index'
        return df.sample(10)
    
    
    def regression_report(y_true, y_pred):
        error = y_true - y_pred
        percentil = [5,25,50,75,95]
        percentil_value = np.percentile(error, percentil)
        
        metrics = [
            ('mean absolute error', mean_absolute_error(y_true, y_pred)),
            ('median absolute error', median_absolute_error(y_true, y_pred)),
            ('mean squared error', mean_squared_error(y_true, y_pred)),
            ('max error', max_error(y_true, y_pred)),
            ('r2 score', r2_score(y_true, y_pred)),
            ('explained variance score', explained_variance_score(y_true, y_pred))
            ]
        
        print('Metrics for regression:')
        for metric_name, metric_value in metrics:
            print(f'{metric_name:>25s}: {metric_value: >20.3f}')
            
        print('\nPercentiles:')
        for p, pv in zip(percentil, percentil_value):
            print(f'{p: 25d}: {pv:>20.3f}')
    
    
    
    @staticmethod
    def get_info_dict(model, y_test, y_pred_test, dataset_name, data_shape, random_state, scaling_choice, test_size, t1, t2, t0, classification):
        scaling_choices = {0: 'None', 1: 'MinMax', 2: 'Standard'} # turns the user input for scaling (int) back into its name
        info = {
            'problem_type': 'classification' if classification else 'regression',
            'model_type': type(model).__name__,
            'dataset_name': dataset_name,
            'num_features': data_shape[1],
            'num_rows': data_shape[0],
            'date': datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),  # Replace hyphens and colons with underscores            
            'random_state': random_state,
            'normalisation': scaling_choices[scaling_choice],
            'test_size': test_size,
            'training_time': t2 - t1,
            'total_time': t2 - t0
        }
        # Get hyperparameters and add them to the info dictionary
        hyperparameters = model.get_params()
        hyperparameters.pop('random_state', None)
        info.update(hyperparameters)

        if classification:
            report_test = classification_report(y_test, y_pred_test, zero_division=1, output_dict=True)
            info['accuracy_or_r2'] = report_test['accuracy']
            for label, metrics in report_test.items():
                if isinstance(metrics, dict):
                    info[f"{label}_precision"] = metrics['precision']
                    info[f"{label}_recall"] = metrics['recall']
                    info[f"{label}_f1_score"] = metrics['f1-score']
                    info[f"{label}_support"] = metrics['support']
                else:
                    info[label] = metrics
        else:
            error = y_test - y_pred_test
            percentil = [5,25,50,75,95]
            percentil_value = np.percentile(error, percentil)
            info['mae'] = mean_absolute_error(y_test, y_pred_test)
            info['median_ae'] = median_absolute_error(y_test, y_pred_test)
            info['rmse'] = mean_squared_error(y_test, y_pred_test, squared=False)
            info['mse'] = mean_squared_error(y_test, y_pred_test)
            info['max_error'] = max_error(y_test, y_pred_test)
            info['accuracy_or_r2'] = r2_score(y_test, y_pred_test)
            info['explained_var_score'] = explained_variance_score(y_test, y_pred_test)
            for p, pv in zip(percentil, percentil_value):
                info[f'p_{p}'] = pv

        info = {k.replace(' ', '_'): v for k, v in info.items()}
        

        
        return info
        
    @staticmethod
    def model_pipeline(model_name, params, dataset_name, dataset_path, test_size, random_state, scaling_choice, classification):
        xgb = False # to avoid error for refencing  before assignment
        if model_name == 'DecisionTreeClassifier':
            model = MyDecisionTreeClassifier()
        elif model_name == 'RandomForestClassifier':
            model = MyRandomForestClassifier()
        elif model_name == 'XGBClassifier':
            xgb = True
            model = MyXGBoostClassifier()
        elif model_name == 'LinearRegression':
            model = MyLinearRegression()
        elif model_name == 'ElasticNetCV':
            model = MyElasticNetCV()
        elif model_name == 'XGBRegressor':
            xgb=True
            model = MyXGBoostRegressor()
        else:
            raise ValueError("Unsupported model type. Please choose from: DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LinearRegression, ElasticNetCV, XGBRegressor")
            
        
        t0 = time.time()
        print("Starting the pipeline...")

        X_train, X_test, y_train, y_test, data_shape = MLToolBox.prepare(
            dataset_path=dataset_path, test_size=test_size, random_state=random_state, 
            feature_scaling=scaling_choice, classification=classification)
        print("\nX_train info: \n", X_train.info())
        print(X_train.head())
        
        
        if classification and xgb:
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)
                    
        t1 = time.time()
        print("Training the model...")
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        t2 = time.time()
        print(f"Training time: {(t2 - t1)//60} minutes, {round((t2 - t1)%60)} seconds")

        if classification:
            print("\n[X_test]:\n", classification_report(y_test, y_pred_test))
        else:
            print("\n[X_test]:\n", MLToolBox.regression_report(y_test, y_pred_test))
        
        print(f"\nTotal execution time: {(t2 - t0)//60} minutes, {round((t2 - t0)%60)} seconds")
        print(MLToolBox.compare_results(y_test, y_pred_test))

        info = MLToolBox.get_info_dict(model=model, data_shape=data_shape, y_test=y_test, y_pred_test=y_pred_test, 
                                       dataset_name=dataset_name, random_state=random_state, scaling_choice=scaling_choice, 
                                       test_size=test_size, t1=t1, t2=t2, t0=t0, classification=classification
                                       )
        
        table = PrettyTable(['Key', 'Value'])
        for key, value in info.items():
            table.add_row([key, value])
        print(table)
        #print(pd.DataFrame([info]).transpose())
        return model, info
        
        
    @staticmethod
    def save_model(model, model_type='other'):
        co2.data.save_model(model, model_type)
        
    
    

    
    
class MLVisualizer:
    @staticmethod
    def confusion_matrix(y_true, y_pred, classes, title=': Decision Tree', save=True, format='png'):
        co2.plot_confusion_matrix(y_true, y_pred, classes, title=title, save=save, format=format)
        
    @staticmethod
    def classification_report(y_true, y_pred):
        co2.styles.display_classification_report(y_true, y_pred)
        
    
    