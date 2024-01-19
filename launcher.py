import os
import argparse
import subprocess
import time
import datetime
import pandas as pd
import sqlite3

from app.models import MLToolBox, DataBaseManager, MyDecisionTreeClassifier
from sklearn.metrics import classification_report

header = """
################################################################
#                          #####  #######  #####               #
#  ####    ##   #####     #     # #     # #     #              #
# #    #  #  #  #    #    #       #     #       #              #
# #      #    # #    #    #       #     #  #####               #
# #      ###### #####     #       #     # #                    #
# #    # #    # #   #     #     # #     # #                    #
#  ####  #    # #    #     #####  ####### #######              #
#                                                              #
#                                                              #
# ###### #    # #  ####   ####  #  ####  #    #  ####          #
# #      ##  ## # #      #      # #    # ##   # #              #
# #####  # ## # #  ####   ####  # #    # # #  #  ####          #
# #      #    # #      #      # # #    # #  # #      #         #
# #      #    # # #    # #    # # #    # #   ## #    #         #
# ###### #    # #  ####   ####  #  ####  #    #  ####          #
#                                                              #
#                                                              #
# #####  #####  ###### #####  #  ####  ##### #  ####  #    #   #
# #    # #    # #      #    # # #    #   #   # #    # ##   #   #
# #    # #    # #####  #    # # #        #   # #    # # #  #   #
# #####  #####  #      #    # # #        #   # #    # #  # #   #
# #      #   #  #      #    # # #    #   #   # #    # #   ##   #
# #      #    # ###### #####  #  ####    #   #  ####  #    #   #
################################################################
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

        
    def get_string_input(self, prompt, default):
        value = input(prompt)
        return value.strip() if value.strip() else default
    
    def get_int_input(self, prompt, default):
        value = input(prompt)
        return int(value) if value.isdigit() else default
    
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

    def manage_docker(self):
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

    def launch_docker_service(self):
        print("Launching the Docker container...")
        command = ['docker', 'run', '-it', '-p', '4000:80', 'auto_co2', 'python3', 'launcher.py']
        process = subprocess.run(command)

        if process.returncode != 0:
            print("An error occurred while launching the Docker container")
        else:
            print("Docker container launched successfully")
            print("The app is now running on http://localhost:4000")
                    
    def fetch_data_from_kaggle(self):
        while True:
            kaggle_auth_file_path = input("Please enter the path to your Kaggle auth file (JSON)")
            if os.path.isfile(kaggle_auth_file_path):
                print("Fetching the data from Kaggle.com...")
                self.run_script('fetch_dataset.py', path=kaggle_auth_file_path)
                break
            else:
                print("Error: The path doesn't lead to a valid file. Please check the path and try again")

        
    def get_countries(self):
        while True:
            country_input = input("Please enter the EU country code(s) you want to keep (separated by a comma, has to be ISO alpha-2 like FR, IT...):\n")
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
                
                
    def get_processed_dataset(self, classification=True):
        if classification:
            dir_path = "../data/processed/classification"
        else:
            dir_path = "../data/processed/regression"
            
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

        random_state = input("Enter random_state (default: 42): ").strip()
        random_state = int(random_state) if random_state.isdigit() else default_random_state

        scaling_choice = input("Enter scaling_choice (0: no scaling, 1: MinMax normalizaion, 2: Standardization, default: 0): ").strip()
        scaling_choice = int(scaling_choice) if scaling_choice in ['0', '1', '2'] else default_scaling_choice

        return test_size, random_state, scaling_choice

    
    def get_classification_model(self):
        while True:
            print("Please select a model:")
            print("[1] Decision Tree")
            print("[2] Random Forest")
            print("[3] XGBoost")
            model_choice = input()
            if model_choice.isdigit() and model_choice in ['1', '2', '3']:
                if model_choice == '1':
                    model = MyDecisionTreeClassifier()
                    params = model.get_parameters(
                        self.get_string_input, self.get_int_input)
                    model.set_params(**params) # Method inherited from sklearn.base.BaseEstimator
                elif model_choice == '2':
                    pass
                elif model_choice == '3':
                    pass
                break
            else:
                print("Error: Invalid choice, please try again")
        return model, model_choice, params
                
     
    def run_classification(self):
        print(os.getcwd())
        dataset_path, dataset_name = self.get_processed_dataset()
        model, model_choice, params = self.get_classification_model()        
        info = self.prepare_and_run_model(model=model, dataset_name=dataset_name, dataset_path=dataset_path, classification=True)
     
        save_model_mapping = {1: 'other', 2: 'other', 3: 'xgb'}
        while True:
            save_choice = input("Do you want to save the model? (y/n)")
            if save_choice in ['y', 'n']:
                if save_choice == 'y':
                    MLToolBox.save_model(model, save_model_mapping[int(model_choice)]) # int >> user input is a string
                break
            else:
                print("Error: Invalid choice, please try again")
        
        while True:
            database_update_choice = input("Do you want to update the database with these results? (y/n)")
            if database_update_choice in ['y', 'n']:
                if database_update_choice == 'y':
                    dbm = DataBaseManager().update_all_tables(info)  # Pass info here
                break
                   
  

    
    def prepare_and_run_model(self, model, dataset_name, dataset_path, classification):  # Changed from model_choice to model
        t0 = time.time()
        test_size, random_state, scaling_choice = self.get_data_treatment()
        
        X_train, X_test, y_train, y_test = MLToolBox.prepare(
            dataset_path=dataset_path, test_size=test_size, random_state=random_state, 
            feature_scaling=scaling_choice, classification=classification)
                
        t1 = time.time()
        print("Training the model...")
        # model.fit(X_train, y_train)  # Removed the line that loads the model
        y_pred_test = model.train_and_predict(X_test, y_test)  # Use train_and_predict method
        t2 = time.time()
        print(f"Training time: {(t2 - t1)//60} minutes, {round((t2 - t1)%60)} seconds")
        
        #y_pred_train = model.predict(X_train)
        
        #print("\n[X_train]:\n", classification_report(y_train, y_pred_train))
        print("\n[X_test]:\n", classification_report(y_test, y_pred_test))
        print(f"\nTotal execution time: {(t2 - t0)//60} minutes, {round((t2 - t0)%60)} seconds")
        print(MLToolBox.compare_results(y_test, y_pred_test))
    
        report_test = classification_report(y_test, y_pred_test, output_dict=True)

        # Create a dictionary with all necessary information
        info = {
            'model_type': type(model).__name__,
            'dataset_name': dataset_name,
            'accuracy_or_r2': report_test['accuracy'],
            'num_features': X_train.shape[1],
            'num_rows': X_train.shape[0],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Add this line
            'problem_type': classification,
            'hyperparameters': model.get_params(),
            'normalisation': scaling_choice,
            'random_state': random_state,
            'test_size': test_size,
            'training_time': t2 - t1,
            'total_time': t2 - t0
        }
        #report_train = classification_report(y_train, y_pred_train, output_dict=True)
        for label, metrics in report_test.items():
            if isinstance(metrics, dict):
                info[f"{label}_precision"] = metrics['precision']
                info[f"{label}_recall"] = metrics['recall']
                info[f"{label}_f1-score"] = metrics['f1-score']
                info[f"{label}_support"] = metrics['support']
            else:
                info[label] = metrics
                
        print(pd.DataFrame([info]).transpose())
        input("Press any key to continue...")  

        return info    
  
    
    
    
    def run_script(self, script_name, args=None, path=None):
        t0 = time.time()
        if path is not None: # For a file possibly outside the current directory
            cmd = ['python', script_name, path]
        else:
            cmd = ['python', script_name]
        if args:
            for flag, value in args.items():
                if isinstance(value, list):
                    cmd.extend([flag] + value) # example: {'--countries': ['FR', 'DE'], '--save': 3}
                else:
                    cmd.extend([flag, str(value)])
        
        subprocess.run(cmd)
        t1 = time.time()
        print(f"Execution time: {(t1 - t0)//60} minutes, {round((t1 - t0)%60)} seconds")
        input("Press any key to continue...")
        
        
        
    def run_sqlite_session(self):
        conn = sqlite3.connect('experiments.db')
        cursor = conn.cursor()

        while True:
            query = input("Enter an SQL query or 'exit' to quit: ")
            if query.lower() == 'exit':
                break

            try:
                cursor.execute(query)
                print(cursor.fetchall())
            except sqlite3.Error as e:
                print(f"An error occurred: {e}")
        
        
 ############################################# MAIN LOOP #############################################
        
    def run(self):
        while True:
            print(header)
            print("Welcome to the Automobile CO2 Emissions Prediction app!")
            print("Please select an option: ")
            print("[1] Restart the app in a Docker container (Docker engine must be running)")
            print("\n########################## CAUTION! ##########################") 
            print("For the following options, you must install the our auto_co2 package first: [ pip install -e . ]")
            print("Please also mind the python libraries in the requirements.txt file...")
            print("Creating a python virual environment is strongly advised!\n")
            print("[2] Fetch the data from Kaggle.com (requires a Kaggle auth file)")
            print("[3] Run preprocessing")
            print("[4] Run a classification model")
            print("[5] Run a regression model")
            print("[8] Exit")
            choice = input('> ')
            
            if choice == "1":
                self.manage_docker()
            
            elif choice == '2':
                self.fetch_data_from_kaggle()
                
            elif choice == '3':
                countries = self.get_countries()
                save = self.get_save_visuals_option()
                preprocessing_choice = self.get_preprocessing_choice()
                if preprocessing_choice == '1':
                    pass
                elif preprocessing_choice == '2':   
                    self.run_script('class_preprocessing.py', args={'--countries': countries, '--save': save})
                elif preprocessing_choice == '3':
                    pass
                elif preprocessing_choice == '4':
                    pass
                
            elif choice == '4':
                self.run_classification()
                
            elif choice == '5':
                pass
            
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