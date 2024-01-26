import os
import argparse
import subprocess
import readline
import time
import datetime
import pandas as pd
import sqlite3
import unittest
from prettytable import PrettyTable

import auto_co2 as co2
from app.models import MLToolBox, MyDecisionTreeClassifier, MyRandomForestClassifier, MyXGBoostClassifier
from app.models import MyLinearRegression, MyElasticNetCV, MyXGBoostRegressor
from app.sqlite_db import DataBaseManager
from sklearn.metrics import classification_report

header = """
##################################################################
#                            #####  #######  #####               #
#    ####    ##   #####     #     # #     # #     #              #
#   #    #  #  #  #    #    #       #     #       #              #
#   #      #    # #    #    #       #     #  #####               #
#   #      ###### #####     #       #     # #                    #
#   #    # #    # #   #     #     # #     # #                    #
#    ####  #    # #    #     #####  ####### #######              #
#                                                                #
#                                                                #
#   ###### #    # #  ####   ####  #  ####  #    #  ####          #
#   #      ##  ## # #      #      # #    # ##   # #              #
#   #####  # ## # #  ####   ####  # #    # # #  #  ####          #
#   #      #    # #      #      # # #    # #  # #      #         #
#   #      #    # # #    # #    # # #    # #   ## #    #         #
#   ###### #    # #  ####   ####  #  ####  #    #  ####          #
#                                                                #
#                                                                #
#   #####  #####  ###### #####  #  ####  ##### #  ####  #    #   #
#   #    # #    # #      #    # # #    #   #   # #    # ##   #   #
#   #    # #    # #####  #    # # #        #   # #    # # #  #   #
#   #####  #####  #      #    # # #        #   # #    # #  # #   #
#   #      #   #  #      #    # # #    #   #   # #    # #   ##   #
#   #      #    # ###### #####  #  ####    #   #  ####  #    #   #
##################################################################
2023-2024
Ludovic CALMETTES
Pierre-Olivier KAPNANG
Matthieu LE NOZACH
Hippolyte NOUYETOWA
"""



class App:
    def __init__(self):
        self.valid_countries = ['ES', 'DE', 'PL', 'IS', 'MT', 'FR', 'BE', 'GR', 'IE', 'NO', 'HR', 'HU', 'SK', 'AT',
                                'PT', 'NL', 'DK', 'LV', 'IT', 'FI', 'CY', 'CZ', 'SE', 'RO', 'SI', 'EE', 'BG', 'LU', 'LT']
        os.chdir('./app') 
        
############################################# PROMPTING TOOLS #############################################

    def parse_input(self, prompt, default_value):
        user_input = input(f"{prompt} (default: {default_value}): ")
        if user_input == '':
            return default_value
        else:
            try: # try to convert to int
                return int(user_input)
            except ValueError:
                try: # if int fails, try to convert to float
                    return float(user_input)
                except ValueError: # if float fails, check for boolean
                    if user_input.lower() in ['true', 'false']:
                        return user_input.lower() == 'true'
                    else: # if not boolean, it's a string
                        return user_input
        
    def get_string_input(self, prompt, default):
        value = input(prompt)
        return value.strip() if value.strip() else default
    
    def get_int_input(self, prompt, default):
        value = input(prompt)
        return int(value) if value.isdigit() else default
    
############################################# END OF PROMPTING TOOLS #############################################

############################################# DOCKER TOOLS #############################################
    def docker_build(self):
        print("Building the Docker image...")
        os.chdir('../')
        command = ['docker', 'build', '--no-cache', '-t', 'auto_co2', '-f', 'Dockerfile', '.']
        process = subprocess.run(command)

        if process.returncode != 0:
            print("An error occurred while building the Docker image")
            input("Press enter to continue...")
        else:
            print("Docker image built successfully")
        os.chdir('./app')

    def manage_docker(self): # Couldn't launch the docker container from the app: pandas error (maybe a conda env activation problem?)
        if os.path.exists('/.dockerenv'):
            print("Error: This app is already running in a Docker container")
            return

        # Check if Docker image exists
        command = ['docker', 'images', '-q', 'auto_co2']
        process = subprocess.run(command, capture_output=True, text=True)

        if process.returncode == 0 and process.stdout.strip():
            print("Docker image already exists")
        else:
            print("Docker image does not exist, creating it...")
            self.docker_build()

        # Check if Docker container is running
        command = ['docker', 'ps', '-q', '-f', 'name=auto_co2']
        process = subprocess.run(command, capture_output=True, text=True)

        if process.returncode == 0 and process.stdout.strip():
            print("Docker container is already running")
        else:
            print("Docker container is not running, launching it...")
            self.launch_docker_service()

    def launch_docker_service(self): # Also not in use
        print("Launching the Docker container...")
        command = ['docker', 'run', '-it', '-p', '4000:80', 'auto_co2', 'python3', 'launcher.py']
        process = subprocess.run(command)

        if process.returncode != 0:
            print("An error occurred while launching the Docker container")
        else:
            print("Docker container launched successfully")
            print("The app is now running on http://localhost:4000")
                    
############################################# END OF DOCKER TOOLS #############################################

############################################# PREPROCESS PROMPTS #############################################
        
    def get_countries(self):
        while True:
            print("Available countries: \n\
               AT, BE, BG, CY, CZ, DE, DK, EE,\n\
               ES, FI, FR, GR, HR, HU, IE, IS,\n\
               IT, LT, LU, LV, MT, NL, NO, PL,\n\
               PT, RO, SE, SI, SK")
            country_input = input("Please enter the EU country code(s) you want to keep, separated by a comma:\n")
            countries = [country.strip().upper() for country in country_input.split(',')]
            if all(country in self.valid_countries for country in countries):
                return countries
            else:
                print("Invalid country, please try again.")
                
                
    def get_save_visuals_option(self):
        while True:
            print("Please select an option:")
            print('[1] Save the tables')
            print("[2] Save the plots")
            print("[3] Save both")
            print("[0] Save none of them")
            save = input()
            if save.isdigit() and save in ['0', '1', '2', '3']:
                return save
            else:
                print("Error: Invalid choice, please try again")
                
                
    def get_preprocessing_choice(self):
        while True:
            print("Which preprocessing do you want to run?")
            print("[1] Preprocessing for data visualization")
            print("[2] Preprocessing for classification (co2 score prediction)")
            print("[3] Preprocessing for regression (co2 emission prediction)")
            print("[4] Preprocessing on aggregated data")
            preprocessing_choice = input()
            if preprocessing_choice.isdigit and preprocessing_choice in ['1', '2', '3', '4']:
                return preprocessing_choice
            else:   
                print("Error: Invalid choice, please try again")
                
############################################# END OF PREPROCESS PROMPTS #############################################

############################################# PRE EXPERIMENT PROMPTS #############################################                
                
    def get_processed_dataset(self, problem_type='classification'):
        if problem_type == 'classification':
            dir_path = "../data/processed/classification"
        elif problem_type == 'regression':
            dir_path = "../data/processed/regression"
        else:
            raise ValueError(f"Invalid problem_type: {problem_type}")   
            
        datasets = os.listdir(dir_path) # List of all the files in the directory
        datasets = sorted(datasets, key=lambda x: os.path.getmtime(os.path.join(dir_path, x)),  reverse=True)[:9] # key parameter takes a function
        
        for i, dataset in enumerate(datasets, start=1):
            print(f"{i}: {dataset}")
        while True:
            dataset_choice = input()
            if dataset_choice.isdigit() and 1 <= int(dataset_choice) <= len(datasets):
                dataset_name = datasets[int(dataset_choice) - 1]
                return os.path.join(dir_path, dataset_name), dataset_name
            else:
                print("Error: Invalid choice, please try again")       
    
    def get_data_treatment(self):
        default_test_size = 0.2
        default_random_state = 42
        default_scaling_choice = 0

        test_size = input("\nEnter test_size (default: 0.2): ").strip()
        test_size = float(test_size) if test_size else default_test_size

        random_state = default_random_state
        user_input = input("Enter random_state (default: 42): ").strip()
        if user_input.isdigit():
            random_state = int(user_input)

        scaling_choice = input("Enter scaling_choice (0: no scaling, 1: MinMax normalizaion, 2: Standardization, default: 0): ").strip()
        scaling_choice = int(scaling_choice) if scaling_choice in ['0', '1', '2'] else default_scaling_choice
        print("random state: " , random_state)
        return test_size, random_state, scaling_choice
    
    def get_hyperparameters(self, model_name):
        # Map model names to their corresponding classes
        model_classes = {
            'DecisionTreeClassifier': MyDecisionTreeClassifier,
            'RandomForestClassifier': MyRandomForestClassifier,
            'XGBClassifier': MyXGBoostClassifier,
            'LinearRegression': MyLinearRegression,
            'ElasticNetCV': MyElasticNetCV,
            'XGBRegressor': MyXGBoostRegressor,
            }
        
        hyperparameters = model_classes[model_name].DEFAULT_HYPERPARAMS
        hps = {}

        print("Press enter to use the hyperparameter's default value.")
        print("[CAUTION] An unexpected value or data type will cause the program to break!")
        print("Please refer to scikit-learn / keras documentations for more information.")

        for hyperparameter, default_value in hyperparameters.items():
            hps[hyperparameter] = self. parse_input(f"Enter {hyperparameter}", default_value)
        return hps

    
    def get_classification_model(self):
        while True:
            print("Please select a model:")
            print("[1] Decision Tree")
            print("[2] Random Forest")
            print("[3] XGBoost")
            model_choice = input()
            if model_choice.isdigit() and model_choice in ['1', '2', '3']:
                if model_choice == '1':
                    model_name = 'DecisionTreeClassifier'
                elif model_choice == '2':
                    # model = MyRandomForestClassifier()
                    model_name = 'RandomForestClassifier'
                    pass
                elif model_choice == '3':
                    # model = MyXGBoostClassifier()
                    model_name = 'XGBClassifier'
                    pass
                break
            else:
                print("Error: Invalid choice, please try again")
        return model_name
    
    
    def get_regression_model(self):
        while True:
            print("Please select a model:")
            print("[1] Linear Regression")
            print("[2] ElasticNet CV")
            print("[3] XGBoost Regressor")
            model_choice = input()
            if model_choice.isdigit() and model_choice in ['1', '2', '3']:
                if model_choice == '1':
                    model_name = 'LinearRegression'
                elif model_choice == '2':
                    model_name = 'ElasticNetCV'
                    pass
                elif model_choice == '3':
                    model_name = 'XGBRegressor'
                    pass
                break
            else:
                print("Error: Invalid choice, please try again")
        return model_name
                

############################################# END OF PRE EXPERIMENT PROMPTS #############################################        
   
############################################# ML PIPELINE ############################################# 
    def run_experiment(self, problem_type):
        print(os.getcwd())
        dataset_path, dataset_name = self.get_processed_dataset(problem_type=problem_type)
        model_name = (self.get_classification_model() if problem_type == 'classification' else self.get_regression_model())
        params = self.get_hyperparameters(model_name)
        test_size, random_state, scaling_choice = self.get_data_treatment()
        ml = MLToolBox()
        model, info = ml.model_pipeline(model_name=model_name, 
                                params=params,
                                dataset_name=dataset_name, 
                                dataset_path=dataset_path,
                                classification=(problem_type == 'classification'), # checks for a true/false
                                test_size=test_size,
                                random_state=random_state,
                                scaling_choice=scaling_choice)
        
        input("Press any key to continue...")   
        #print(f"info after model_pipeline: {info}") debug print
    
        self.save_model(model, model_name)
        self.update_database(info, problem_type)
        
        
    def run_script(self, script_name, args=None, path=None):
        t0 = time.time()
        if path is not None: # For a file possibly outside the current directory
            cmd = ['python', script_name, path]
        else:
            cmd = ['python', script_name]
        if args:
            for flag, value in args.items():
                if isinstance(value, list):
                    cmd.extend([flag] + value) 
                else:
                    cmd.extend([flag, str(value)])
        
        subprocess.run(cmd)
        t1 = time.time()
        print(f"Execution time: {(t1 - t0)//60} minutes, {round((t1 - t0)%60)} seconds")
        input("Press any key to continue...")
        
############################################# END OF ML PIPELINE #############################################

############################################# POST EXPERIMENT PROMPTS #############################################     

    def save_model(self, model, model_name):
        save_model_mapping = {1: 'other', 2: 'other', 3: 'xgb'}
        while True:
            save_choice = input("Do you want to save the model? (y/n)")
            if save_choice in ['y', 'n']:
                if save_choice == 'y':
                    MLToolBox.save_model(model, save_model_mapping[int(model_name)]) # int >> user input is a string        
                break
            else:
                print("Error: Invalid choice, please try again")
                
    def update_database(self, info, problem_type):
        while True:
            database_update_choice = input("Do you want to update the database with these results? (y/n)")
            if database_update_choice in ['y', 'n']:
                if database_update_choice == 'y':
                    #print(f"info before sql_pipeline: {info}") #debug print
                    db_path = '../database/experiments.db'  # replace with your actual db_path
                    dbm = DataBaseManager().sql_pipeline(info=info, db_path=db_path, classification=(problem_type == 'classification'))
                    print("Experiment saved successfully!")
                    input("Press any key to continue...")
                break
            
############################################# END OF POST EXPERIMENT PROMPTS #############################################
        
    def run_sqlite_session(self):
        conn = sqlite3.connect('../database/experiments.db')
        cur = conn.cursor()

        with open('query_logs.txt', 'a+') as f:
            f.seek(0)
            lines = f.readlines()[:5]
            print("Last 5 queries:")
            for line in lines:
                print(line.strip())

        while True:
            query = input("Enter an SQL query or 'exit' to quit: ")
            readline.add_history(query)  # Add this line to save the input to the readline history

            if query.lower() == 'exit':
                break

            try:
                cur.execute(query)
                results = cur.fetchall()
                column_names = [description[0] for description in cur.description]
                
                num_columns_per_table = 8  # Splitting tables with too many columns...
                for start_col in range(0, len(column_names), num_columns_per_table):
                    end_col = start_col + num_columns_per_table
                    table = PrettyTable(column_names[start_col:end_col])
                    for row in results:
                        table.add_row(row[start_col:end_col])

                    print(table)

                with open('query_logs.txt', 'a') as f:
                    f.write(query + '\n')

            except sqlite3.Error as e:
                print(f"An error occurred: {e}")
        
 ############################################# MAIN LOOP #############################################
        
    def run(self):
        while True:
            print(header)
            print("Welcome to the Automobile CO2 Emissions Prediction app!")
            print("Please select an option: ")
            print("[1] Build the repo's Docker image, linux/docker engine users only, docker engine must be running")
            print("Instructions for Windows/Mac & docker_desktop in the readme.md")
            print("\n########################## CAUTION! ##########################") 
            print("For the following options, you must install the our auto_co2 package first: [ pip install -e . ]")
            print("Please also mind the python libraries in the requirements.txt file...")
            print("Creating a python virual environment is strongly advised!\n")
            print("[2] Fetch the dataset from github.com")
            print("[3] Run preprocessing")
            print("[4] Run a classification model")
            print("[5] Run a regression model")
            print("[8] Run a SQLite session")
            print("[9] Exit")
            choice = input('> ')
            
            if choice == "1":
                self.docker_build()
            
            elif choice == '2':
                co2.data.download_file()
                
            elif choice == '3':
                countries = self.get_countries()
                save = self.get_save_visuals_option()
                preprocessing_choice = self.get_preprocessing_choice()
                if preprocessing_choice == '1':
                    pass
                elif preprocessing_choice == '2':   
                    self.run_script('class_preprocessing.py', args={'--countries': countries, '--save': save})
                elif preprocessing_choice == '3':
                    self.run_script('reg_preprocessing.py', args={'--countries': countries, '--save': save})
                elif preprocessing_choice == '4':
                    pass
                
            elif choice == '4':
                self.run_experiment('classification')
                
            elif choice == '5':
                self.run_experiment('regression')
            
            elif choice == '8':
                self.run_sqlite_session()
            
            elif choice == '9':
                break
            
            else:
                print("Error: Invalid choice. Please try again")
                continue

############################################# END OF MAIN LOOP #############################################
            

        
        
        

            
            
            
if __name__ == '__main__':
    App().run()