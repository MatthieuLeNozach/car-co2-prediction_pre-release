import os
import sys
sys.path.insert(0, '../src/')
from pathlib import Path
import sqlite3
from sqlite3 import Error

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

import auto_co2 as co2

########## CLASSIFICATION MODELS ##########


class MyDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, criterion='entropy', 
                 max_depth=20, min_samples_leaf=1, 
                 random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0):
        super().__init__(criterion=criterion, 
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease)
        
    def get_parameters(self, get_string_input, get_int_input):
        params = {}
        params['criterion'] = get_string_input("Enter criterion (default: 'gini'):", 'gini')
        params['max_depth'] = get_int_input("Enter max depth (default: None):", None)
        params['min_samples_leaf'] = get_int_input("Enter min samples leaf (default: 1):", 1)
        params['max_leaf_nodes'] = get_int_input("Enter max leaf nodes (default: None):", None)
        return params
                     
    def train_and_predict(self, X, y):
        self.fit(X, y)
        y_pred =  self.predict(X)
        return y_pred
    

        
    def feature_importances(self, data):
        co2.styles.display_feature_importances(self, data)

                 
            
                 
class MLToolBox:  

    @staticmethod
    def load_data(filepath, separate_Xy=True, classification=False):
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.pkl'):
            data = pd.read_pickle(filepath)
        else:
            raise ValueError("Unsupported file type. Please provide a .csv or .pkl file.")

        print(data.info())

        if separate_Xy:
            return MLToolBox.separate_target(data, classification)
        else:
            return data

    @staticmethod
    def separate_target(data, classification):
        if classification:
            X = data.drop(columns=['Co2Grade'])
            y = data['Co2Grade']
        else:
            X = data.drop(columns=['Co2EmissionsWltp'])
            y = data['Co2EmissionsWltp']
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
        X, y = MLToolBox.load_data(dataset_path, separate_Xy=True, classification=classification)
        print(f"Loaded {dataset_path}")
            
        X_train, X_test, y_train, y_test = MLToolBox.split_data(X, y, test_size=test_size, random_state=random_state)
        print(f"Data has been split into train and test sets with a test size of {test_size}, random state: {random_state}")
        
        if feature_scaling == 1: # minmax normalisation
            X_train = MLToolBox.minmax_scale(X_train)
            X_test = MLToolBox.minmax_scale(X_test)
            print("Data has been normalized")    
        elif feature_scaling == 2: #standardisation
            X_train = MLToolBox.standard_scale(X_train)
            X_test = MLToolBox.standard_scale(X_test)
            print("Data has been standardized")
        else:
            print("No feature scaling applied")

        return X_train, X_test, y_train, y_test
    
    
    @staticmethod
    def compare_results(y_true, y_pred):
        df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        df.index.name = 'Row Index'
        return df.sample(10)
        
        
        
    @staticmethod
    def save_model(model, model_type='other'):
        co2.data.save_model(model, model_type)
        
     
    
    
    
    
    
class DataBaseManager:
    DB_DIR = '../database'
    DB_PATH = '../database/experiments.db'
    
    
    EXPERIMENTS_COLUMNS = ['id', 'model_type', 'dataset_name', 'date', 'problem_type', 
                           'num_features', 'num_rows', 'accuracy_or_r2', 'random_state', 
                           'normalisation', 'test_size', 'training_time', 'total_time']

    EXPERIMENTS_SQL_INSTRUCTIONS = ['integer PRIMARY KEY AUTOINCREMENT', 'text NOT NULL', 'text NOT NULL', 
                                    'text', 'text NOT NULL', 'integer NOT NULL', 'integer NOT NULL', 
                                    'real NOT NULL', 'integer NOT NULL', 'integer NOT NULL', 'real NOT NULL', 
                                    'real NOT NULL', 'real NOT NULL']

    XP_HYPERPARAMETERS_COLUMNS = ['id', 'model_type', 'problem_type', 'criterion', 'max_depth', 
                                  'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 
                                  'rf_n_estimators', 'rf_max_features', 'rf_bootstrap', 
                                  'xgb_learning_rate', 'xgb_gamma', 'lr_fit_intercept', 
                                  'lr_normalize', 'lr_copy_x', 'lr_n_jobs', 'encv_l1_ratio', 
                                  'encv_eps', 'encv_n_alphas', 'encv_fit_intercept']

    XP_HYPERPARAMETERS_SQL_INSTRUCTIONS = ['integer', 'text NOT NULL', 'text NOT NULL', 'text', 
                                        'integer', 'integer', 'integer', 'real', 'integer', 'text', 'text', 
                                        'real', 'real', 'text', 'text', 'text', 'integer', 'real', 
                                        'real', 'integer', 'text', 
                                        'FOREIGN KEY (id) REFERENCES experiments (id)']

    CLASS_RESULTS_COLUMNS = ['id', 'model_type', 'metric', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

    CLASS_RESULTS_SQL_INSTRUCTIONS = ['integer', 'text NOT NULL', 'text NOT NULL', 
                                    'real', 'real', 'real', 'real', 'real', 'real', 'real', 
                                    'FOREIGN KEY (id) REFERENCES experiments (id)']

    MODEL_HYPERPARAMETERS_COLUMNS = ['model_type', 'hyperparameter1', 'hyperparameter2', 
                                     'hyperparameter3', 'hyperparameter4']

    MODEL_HYPERPARAMETERS_SQL_INSTRUCTIONS = ['text PRIMARY KEY', 'text NOT NULL', 'text NOT NULL', 
                                              'text NOT NULL', 'text NOT NULL']
    
    MODEL_TO_HYPERPARAMETERS = {
        'DecisionTreeClassifier': ['criterion', 'max_depth', 'min_samples_leaf', 'max_leaf_nodes'],
        'RandomForestClassifier': ['n_estimators', 'criterion', 'max_depth', 'max_features'],
        'XGBClassifier': ['learning_rate', 'max_depth', 'n_estimators', 'gamma'],
        'LinearRegression': ['fit_intercept', 'normalize', 'copy_X', 'positive'],
        'ElasticNetCV': ['l1_ratio', 'eps', 'n_alphas', 'fit_intercept'],
        'XGBRegressor': ['learning_rate', 'max_depth', 'n_estimators', 'gamma']
        }

    
    @staticmethod
    def convert_int_to_float(values):
        for i, value in enumerate(values):
            if isinstance(value, int):
                print(f"Converting value {value} at index {i} to float.")
                values[i] = float(value)
            elif not isinstance(value, (float, str)):
                print(f"Error: Unsupported data type {type(value)} at index {i}.")
        return values

    
    @staticmethod
    def db_connection(db_path):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            print(f"Connection to SQLite DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")
        return conn
    
    @staticmethod
    def create_table_sql(table_name, columns, sql_instructions):
        column_defs = ', '.join(f'{col} {sql}' for col, sql in zip(columns, sql_instructions))
        return f'CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})'
        
    def db_init(db_path):
        if not os.path.exists(DataBaseManager.DB_DIR):
            os.makedirs(DataBaseManager.DB_DIR)
        
        conn = DataBaseManager.db_connection(db_path)  # Use db_path here
        c = conn.cursor()  # Create cursor here

        c.execute('PRAGMA foreign_keys = ON;')  # Enable foreign key constraints

        try:
            tables_and_columns = {
                'experiments': (DataBaseManager.EXPERIMENTS_COLUMNS, DataBaseManager.EXPERIMENTS_SQL_INSTRUCTIONS),
                'xp_hyperparameters': (DataBaseManager.XP_HYPERPARAMETERS_COLUMNS, DataBaseManager.XP_HYPERPARAMETERS_SQL_INSTRUCTIONS),
                'class_results': (DataBaseManager.CLASS_RESULTS_COLUMNS, DataBaseManager.CLASS_RESULTS_SQL_INSTRUCTIONS),
                'model_hyperparameters': (DataBaseManager.MODEL_HYPERPARAMETERS_COLUMNS, DataBaseManager.MODEL_HYPERPARAMETERS_SQL_INSTRUCTIONS)
            }

            for table_name, (columns, sql_instructions) in tables_and_columns.items():
                print(f"Creating {table_name} table...")
                c.execute(DataBaseManager.create_table_sql(table_name, columns, sql_instructions))
            print("Tables created!")
        except Error as e:
            print(e)

    @staticmethod
    def add_table(conn, info, table_name, keys_to_add):
        cur = conn.cursor()
        print(f"Adding data to {table_name} table...")
        for key in keys_to_add: # Ensure all necessary columns exist
            try:
                cur.execute(f"SELECT {key} FROM {table_name}")
            except sqlite3.OperationalError:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {key}")
                
        # Insert the data
        keys = ', '.join(keys_to_add)
        placeholders = ', '.join('?' for _ in keys_to_add)
        values = tuple(info.get(key) for key in keys_to_add)
        values = DataBaseManager.convert_int_to_float(list(values))  # Convert integers to floats
        sql = f"INSERT INTO {table_name} ({keys}) VALUES ({placeholders})"
        cur.execute(sql, values)
        print(f"Data added to {table_name}!")
        return cur.lastrowid


    @staticmethod
    def add_experiment(conn, info):
        print(f"Debug: info = {info}")  # Debugging print statement

        keys_to_add = DataBaseManager.EXPERIMENTS_COLUMNS  # Use the constant here

        experiment_id = DataBaseManager.add_table(conn, info, 'experiments', keys_to_add)
        return experiment_id  # return the experiment_id

    @staticmethod 
    def add_xp_hyperparameters(conn, info):
        print(f"Debug: info = {info}")  # Debugging print statement

        keys_to_add = DataBaseManager.XP_HYPERPARAMETERS_COLUMNS  # Use the constant here
        
        return DataBaseManager.add_table(conn, info, 'xp_hyperparameters', keys_to_add)

    @staticmethod
    def add_class_results(conn, info):
        print(f"Debug: info = {info}")  # Debugging print statement

        print("Adding class results to database...")
        keys_to_add = DataBaseManager.CLASS_RESULTS_COLUMNS  # Use the constant here
        metrics = ['precision', 'recall', 'f1-score', 'support']
        for metric in metrics:
            data = {'model_type': info['model_type'], 'metric': metric}
            for class_name in DataBaseManager.CLASS_RESULTS_COLUMNS[-7:]:  # Iterate over the last 7 elements
                if f"{class_name}_{metric}" in info:
                    data[class_name] = info[f"{class_name}_{metric}"]
            DataBaseManager.add_table(conn, data, 'class_results', keys_to_add)
        print("Class results added!")
         

    @staticmethod
    def update_all_tables(info, db_path='../database/experiments.db'):
        print("Checking if database exists...")
        DataBaseManager.db_init(db_path)
        print("Initializing database...")
        conn = DataBaseManager.db_connection(db_path)
        print("Connection established, updating tables...")
        print("Updating experiments table...")
        experiment_id = DataBaseManager.add_experiment(conn, info)
        print("Experiment added")
        print("Updating xp_hyperparameters table...")
        hyperparameters_id = DataBaseManager.add_xp_hyperparameters(conn, info)
        print("XP hyperparameters added")
        print("Updating class_results table...")
        class_results_id = DataBaseManager.add_class_results(conn, info)
        print("Class results added")
        conn.commit()
        print("Changes committed!")
        return experiment_id, hyperparameters_id, class_results_id
    
    
class MLVisualizer:
    @staticmethod
    def confusion_matrix(y_true, y_pred, classes, title=': Decision Tree', save=True, format='png'):
        co2.plot_confusion_matrix(y_true, y_pred, classes, title=title, save=save, format=format)
        
    @staticmethod
    def classification_report(y_true, y_pred):
        co2.styles.display_classification_report(y_true, y_pred)
        
    
    