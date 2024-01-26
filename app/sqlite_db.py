import os
import sys
sys.path.insert(0, '../src/')
from pathlib import Path
import sqlite3
from sqlite3 import Error

import numpy as np
import pandas as pd


import auto_co2 as co2
from app.models import MyDecisionTreeClassifier, MyRandomForestClassifier, MyXGBoostClassifier

import sqlite3
from sqlite3 import Error

class DataBaseManager:
    DB_DIR = '../database'
    DB_PATH = '../database/experiments.db'
    
    XP_HYPERPARAMETERS_COLUMNS = {
        'model_type': 'TEXT', 'problem_type': 'TEXT', 
        'criterion': 'TEXT', 'max_depth': 'INTEGER', 'min_samples_leaf': 'INTEGER', 
        'max_leaf_nodes': 'INTEGER', 'min_impurity_decrease': 'REAL', 'n_estimators': 'INTEGER', 
        'max_features': 'TEXT', 'bootstrap': 'BOOLEAN', 'learning_rate': 'REAL', 
        'gamma': 'REAL', 'fit_intercept': 'BOOLEAN', 'normalize': 'BOOLEAN', 
        'copy_x': 'BOOLEAN', 'n_jobs': 'INTEGER', 'l1_ratio': 'REAL', 
        'eps': 'REAL', 'n_alphas': 'INTEGER', 'min_samples_split': 'INTEGER', 
        'objective': 'TEXT', 'lambda': 'REAL'
        }
    ID_AND_XP_HYPERPARAMETERS_COLUMNS = {'id': 'INTEGER PRIMARY KEY', **XP_HYPERPARAMETERS_COLUMNS}
   
    LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    CLASS_RESULTS_COLUMNS = {
        'row_id': 'INTEGER PRIMARY KEY', 'id': 'INTEGER', 'metric': 'TEXT', 
        'model_type': 'TEXT', 'weighted_avg': 'REAL', 'macro_avg': 'REAL'
    }
    for label in LABELS:
        CLASS_RESULTS_COLUMNS[label] = 'REAL'
        
    REG_RESULTS_COLUMNS = {
        'model_type': 'TEXT', 'r2': 'REAL', 'mse': 'REAL', 
        'mae': 'REAL', 'rmse': 'REAL', 'median_ae': 'REAL', 'explained_var_score': 'REAL',
        'max_error': 'REAL', 'p_5': 'REAL', 'p_25': 'REAL', 'p_50': 'REAL',
        'p_75': 'REAL', 'p_95': 'REAL'
        }
    ID_AND_REG_RESULTS_COLUMNS = {'id': 'INTEGER PRIMARY KEY', **REG_RESULTS_COLUMNS} 

    EXPERIMENTS_COLUMNS = {
         'problem_type': 'TEXT', 'model_type': 'TEXT', 
         'dataset_name': 'TEXT', 'date': 'TEXT', 'num_features': 'INTEGER', 
         'num_rows': 'INTEGER', 'accuracy_or_r2': 'REAL', 'random_state': 'INTEGER' ,
         'normalisation': 'TEXT', 'test_size': 'REAL', 'training_time': 'REAL', 
         'total_time': 'REAL'
         }
    ID_AND_EXPERIMENTS_COLUMNS = {'id': 'INTEGER PRIMARY KEY', **EXPERIMENTS_COLUMNS}
   
   
    MASTER_TABLE_COLUMNS = {
        **EXPERIMENTS_COLUMNS,
        **XP_HYPERPARAMETERS_COLUMNS,
        **CLASS_RESULTS_COLUMNS,
        **REG_RESULTS_COLUMNS
        }
    ID_AND_COLUMNS_MASTER_TABLE = {'id': 'integer PRIMARY KEY', **MASTER_TABLE_COLUMNS}
    
   
    MODEL_HYPERPARAMETERS_COLUMNS = {
        'model_type': 'TEXT PRIMARY KEY', 
        'hp_1': 'TEXT', 'hp_2': 'TEXT', 'hp_3': 'TEXT', 
        'hp_4': 'TEXT', 'hp_5': 'TEXT', 'hp_6': 'TEXT'
        }
    
    MODEL_TO_HYPERPARAMETERS = {
        'DecisionTreeClassifier': ['criterion', 'max_depth', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight'],
        'RandomForestClassifier': ['criterion', 'max_depth', 'n_estimators', 'max_features', 'max_leaf_nodes', 'bootstrap'],
        'XGBClassifier': ['learning_rate', 'max_depth', 'n_estimators', 'gamma', 'lambda', 'objective'],
        'LinearRegression': ['fit_intercept', 'normalize', 'copy_X', 'n-jobs'],
        'ElasticNetCV': ['l1_ratio', 'eps', 'cv', 'fit_intercept', 'max_iter', 'tol'],
        'XGBRegressor': ['max_depth', 'learning_rate', 'n_estimators', 'gamma', 'reg_alpha', 'reg_lambda'],
        }

    #MODEL_CLASSES = [MyDecisionTreeClassifier, MyRandomForestClassifier, MyXGBoostClassifier]
    

    @staticmethod
    def convert_int_to_float(data):
        return {key: float(value) if isinstance(value, int) else value for key, value in data.items()}
    
    @staticmethod
    def transform_hyphens(info):
        return {key.replace('-', '_'): value for key, value in info.items()}
    

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
    def db_init(db_path):
        if not os.path.exists(DataBaseManager.DB_DIR):
            os.makedirs(DataBaseManager.DB_DIR)
        
        conn = DataBaseManager.db_connection(db_path)
        c = conn.cursor()

        c.execute('PRAGMA foreign_keys = ON;')

        # Generate the SQL command to create the table
        create_table_req = DataBaseManager.create_table_request('master', 
            DataBaseManager.ID_AND_COLUMNS_MASTER_TABLE)
        try:
            print("Creating master table...")
            c.execute(create_table_req)
            print("Table created!")
        except Error as e:
            print(e) 

    @staticmethod
    def check_and_init_db(db_path):
        # Check if database exists and initialize it if it doesn't
        if not os.path.exists(db_path):
            DataBaseManager.db_init(db_path)  

    @staticmethod
    def create_table_request(table_name, columns_dict):
        column_defs = ', '.join(f'{col} {sql}' for col, sql in columns_dict.items())
        q = f"""
        CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})
        """
        return q
    
    @staticmethod
    def check_if_table_exists(cur, table_name):
        # Check if the table exists
        q = f"""
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' AND name=?
        """
        cur.execute(q, (table_name,))
        result = cur.fetchone()
        return result is not None        # Return True if the table exists, False otherwise


    @staticmethod
    def create_table_if_not_exists(cur, table_name, columns_dict):
        if not DataBaseManager.check_if_table_exists(cur, table_name):
            print(f"Creating {table_name} table...")
            create_table_request = DataBaseManager.create_table_request(table_name, columns_dict)
            print(f"Executing: {create_table_request}")  # Print the CREATE TABLE statement
            cur.execute(create_table_request)
            print("Table created!")


    @staticmethod
    def add_columns_to_table(cur, table_name, columns_dict):
        for column, type in columns_dict.items():
            try:
                q = f"""
                SELECT {column}
                FROM {table_name}
                """
                cur.execute(q)
            except sqlite3.OperationalError:
                q = f"""
                ALTER TABLE {table_name}
                ADD COLUMN {column} {type}
                """
                cur.execute(q)

    @staticmethod
    def insert_into_table(cur, table_name, data_dict):
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join('?' * len(data_dict))
        sql = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
        cur.execute(sql, list(data_dict.values()))
        cur.connection.commit()
        
    @staticmethod
    def select_last_row(cur):
        q = f"""
        SELECT *
        FROM master
        WHERE id = (
            SELECT MAX(id)
            FROM master
            )
        """
        cur.execute(q)
        return cur.fetchone()    
    
    @staticmethod
    def get_last_row_as_dict(cur):
        q = """
        PRAGMA table_info(master);
        """
        cur.execute(q)  
        columns = [column[1] for column in cur.fetchall()]
        column_indices = {column: index for index, column in enumerate(columns)}
        row = DataBaseManager.select_last_row(cur)

        
        if row is None:
            return None
        
        # print("select last row from get_last_row_as_dict", row) debug print
        return {col: row[column_indices[col]] for col in columns if col in column_indices}
            
    @staticmethod
    def add_experiment_to_master(db_path, info):
        #print(f"Debug: info = {info}")
        info = DataBaseManager.transform_hyphens(info) # Replace hyphens with underscores
        #print("COLUMNS...", DataBaseManager.ID_AND_COLUMNS_MASTER_TABLE)
        DataBaseManager.check_and_init_db(db_path)   # Check and initialize database if necessary

        conn = DataBaseManager.db_connection(db_path)
        cur = conn.cursor()
        
        keys_to_add = DataBaseManager.ID_AND_COLUMNS_MASTER_TABLE
        #print("Adding data to master table...", DataBaseManager.ID_AND_COLUMNS_MASTER_TABLE)

        DataBaseManager.create_table_if_not_exists(cur, 'master', keys_to_add)
        
        # Add new columns from info dictionary
        info_columns = {key: 'TEXT' for key in info.keys()} # Assuming all new columns are of type 'TEXT'
        DataBaseManager.add_columns_to_table(cur, 'master', info_columns)

        # Get the last row and calculate the id value
        last_row = DataBaseManager.get_last_row_as_dict(cur)
        id_value = (last_row['id'] + 1) if last_row else 1
        
        #print("\n info just before inserting into master table...", info)
        values = {key: info.get(key) for key in info.keys()} # Prepare values for insertion, needs to be a dictionary
        #print("Values to be inserted into master table...", values) debug print
        #values = DataBaseManager.convert_int_to_float(values)

        # Add the calculated id value to the values dictionary
        values['id'] = id_value

        DataBaseManager.insert_into_table(cur, 'master', values)

        cur.connection.commit()
        print("Data added to master!")
        return id_value
        
    
    @staticmethod
    def create_model_hyperparameters_table(cur):
        q = f"""
        CREATE TABLE IF NOT EXISTS model_hyperparameters (
            model TEXT PRIMARY KEY,
            hp1 TEXT,
            hp2 TEXT,
            hp3 TEXT, 
            hp4 TEXT,
            hp5 TEXT,
            hp6 TEXT
        )
        """
        cur.execute(q)

    @staticmethod
    def initialize_model_hyperparameters(cur):
        # Drop the table if it exists
        q = f"""
        DROP TABLE IF EXISTS model_hyperparameters
        """
        cur.execute(q)

        # Create the table
        DataBaseManager.create_model_hyperparameters_table(cur)

        # Insert the models and their hyperparameters {model: [hp1,hp2...]}
        for model, hyperparameters in DataBaseManager.MODEL_TO_HYPERPARAMETERS.items():
            hyperparameters = list(hyperparameters) + ([None] * (6 - len(hyperparameters))) # 6 must be changed # of hps is modified!
            q = f"""
            INSERT INTO model_hyperparameters (model, hp1, hp2, hp3, hp4, hp5, hp6) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """ # Change # of hps and '?' manually above for different number of hps...
            cur.execute(q, (model, *hyperparameters))
            print(f"Added {model} to model_hyperparameters table")
        cur.connection.commit()
           
    @staticmethod
    def split_master_into_class_table(db_path):
        conn = DataBaseManager.db_connection(db_path)
        cur = conn.cursor()
        row = DataBaseManager.get_last_row_as_dict(cur)

        DataBaseManager.create_table_if_not_exists(cur, 'class_results', 
                                                DataBaseManager.CLASS_RESULTS_COLUMNS, 
                                                )
            
        id = row['id']
        model_type = row['model_type']
        metrics = ['precision', 'recall', 'f1_score', 'support']
        class_results_data = []
        for metric in metrics:
            data = {
                'id': id,
                'model_type': model_type,
                'metric': metric,
                'weighted_avg': row[f'weighted_avg_{metric}'],
                'macro_avg': row[f'macro_avg_{metric}'],
            }
            for label in DataBaseManager.LABELS:
                data[label] = row[f'{label}_{metric}']
            class_results_data.append(data)
            
            # Insert the data into the class_results table
        for data in class_results_data:
            DataBaseManager.insert_into_table(cur, 'class_results', data)

        cur.connection.commit()
        
        
    @staticmethod
    def split_master_into_reg_table(db_path):
        conn = DataBaseManager.db_connection(db_path)
        cur = conn.cursor()
        row = DataBaseManager.get_last_row_as_dict(cur)
        
        DataBaseManager.create_table_if_not_exists(cur, 'reg_results', DataBaseManager.ID_AND_REG_RESULTS_COLUMNS)
        
        reg_results_data = {col: row[col] for col in DataBaseManager.ID_AND_REG_RESULTS_COLUMNS if col in row}
        if 'accuracy_or_r2' in row:
            reg_results_data['r2'] = row['accuracy_or_r2']
            
        DataBaseManager.insert_into_table(cur, 'reg_results', reg_results_data)
        
        cur.connection.commit()
                
    @staticmethod
    def split_master_into_general_tables(db_path):
        conn = DataBaseManager.db_connection(db_path)
        cur = conn.cursor()
        
        DataBaseManager.initialize_model_hyperparameters(cur)
        
        DataBaseManager.create_table_if_not_exists(cur, 'experiments', 
                                                DataBaseManager.ID_AND_EXPERIMENTS_COLUMNS, 
                                                )
        
        DataBaseManager.create_table_if_not_exists(cur, 'xp_hyperparameters',
                                                    DataBaseManager.ID_AND_XP_HYPERPARAMETERS_COLUMNS,
                                                    )
        
        
        row = DataBaseManager.get_last_row_as_dict(cur)
        print("Printing row from split_master_into_general_tables", row)
        
        experiment_data = {col: row[col] for col in DataBaseManager.ID_AND_EXPERIMENTS_COLUMNS if col in row}
        hyperparameter_xp_data = {col: row[col] for col in DataBaseManager.ID_AND_XP_HYPERPARAMETERS_COLUMNS if col in row}

        DataBaseManager.insert_into_table(cur, 'experiments', experiment_data)
        DataBaseManager.insert_into_table(cur, 'xp_hyperparameters', hyperparameter_xp_data)
            
    @staticmethod
    def sql_pipeline(info, db_path=None, classification=False):
        # triggers every step of the pipeline:
        # 1. adds experiment to master table
        # 2. splits master table into general tables
        # 3. splits master table into class results table if classification
        # 4. splits master table into reg results table if regression
        
        if db_path is None:
            db_path = DataBaseManager.DB_PATH
        
        id = DataBaseManager.add_experiment_to_master(db_path, info)
        DataBaseManager.split_master_into_general_tables(db_path)
        if classification:
            DataBaseManager.split_master_into_class_table(db_path)
        else:
            DataBaseManager.split_master_into_reg_table(db_path)
        return id
    