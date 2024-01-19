import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import zipfile
import datetime
from xgboost import XGBClassifier, XGBRegressor
import kaggle
import pickle
from tensorflow.keras.models import Model


########## Constants ##########
COL_NAMES_SHORTFORM = [
        'ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk',
        'Cn', 'Ct', 'Cr', 'r', 'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)',
        'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'IT',
        'Ernedc (g/km)', 'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration',
        'Fuel consumption ', 'Electric range (km)'
        ]

COL_NAMES_LONGFORM = [ # Official column names from EEA's' website
        'ID', 'Country', 'VehicleFamilyIdentification', 'Pool', 'ManufacturerName', 'ManufNameOem',
        'ManufNameMS', 'TypeApprovalNumber', 'Type', 'Variant', 'Version', 'Make', 'CommercialName',
        'VehicleCategory', 'CategoryOf', 'TotalNewRegistrations', 'MassRunningOrder',
        'WltpTestMass', 'Co2EmissionsNedc', 'Co2EmissionsWltp',
        'BaseWheel', 'AxleWidthSteering', 'AxleWidthOther', 'FuelType', 'FuelMode',
        'EngineCapacity', 'EnginePower', 'ElectricConsumption',
        'InnovativeTechnology', 'InnovativeEmissionsReduction',
        'InnovativeEmissionsReductionWltp', 'DeviationFactor', 'VerificationFactor',
        'Status', 'RegistrationYear', 'RegistrationDate', 'FuelConsumption', 'ElectricRange'
        ]


########## End of Constants ##########



########## Security tools ##########

def secure_path(filepath):
    root = Path('../')
    absolute_filepath = Path(filepath).resolve()
    
    if root not in absolute_filepath.parents:
        raise ValueError(f"Path {filepath} is not within the repository root, please save the file within the repository boundaries and try again.")


########## End of security tools ##########




########## Fetching raw data from Kaggle ##########

def download_co2_data(auth_file_path, filepath='../data/raw'):
    dataset_id = 'matthieulenozach/automobile-co2-emissions-eu-2021'
    dataset_name = dataset_id.split('/')[-1]  # Get the dataset name
    zipfile_path = os.path.join(filepath, f"{dataset_name}.zip")
    
    if os.path.isfile(zipfile_path):
        print(f"File {zipfile_path} already exists.")
        return zipfile_path

    # Save the original KAGGLE_CONFIG_DIR
    original_kaggle_config_dir = os.environ.get('KAGGLE_CONFIG_DIR')
    # Point KAGGLE_CONFIG_DIR to the directory containing kaggle.json
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(auth_file_path)

    os.system(f'kaggle datasets download -d {dataset_id} -p {filepath}')

    # Restore the original KAGGLE_CONFIG_DIR
    if original_kaggle_config_dir is not None:
        os.environ['KAGGLE_CONFIG_DIR'] = original_kaggle_config_dir
    else:
        del os.environ['KAGGLE_CONFIG_DIR']

    print(f"{zipfile_path} has been downloaded successfully.")
    return zipfile_path


def load_co2_data(dataset_name="automobile-co2-emissions-eu-2021", filepath='../data/raw'):
    zip_filename = f"{dataset_name}.zip"
    csv_filename = "auto_co2_eur_21_raw.csv"
    csv_file_path = os.path.join(filepath, csv_filename)
    zip_file_path = os.path.join(filepath, zip_filename)

    if os.path.isfile(csv_file_path):
        data = pd.read_csv(csv_file_path, low_memory=False)
        return data
    elif os.path.isfile(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(filepath)
        if os.path.isfile(csv_file_path):
            data = pd.read_csv(csv_file_path, low_memory=False)
            return data
        else:
            print(f"Error: The CSV file ({csv_file_path}) was not extracted from the ZIP file.")
            return None
    else:
        print(f"Error: Neither the CSV file ({csv_file_path}) nor the ZIP file ({zip_file_path}) exists.")
        return None


def download_and_load_co2_data(auth_file_path, filepath='../data/raw'):
    dataset_name = download_co2_data(auth_file_path, filepath)
    data = load_co2_data(dataset_name, filepath)
    return data

########## End of fetching raw data from Kaggle  ##########




########## Loading processed data ##########
 
def load_processed_helper(classification=False, filepath='../data/processed', filename=None):
    if filename is None:
        if classification:
            filepath = os.path.join(filepath, 'classification')
            filename = 'co2_classification'
        else:
            filepath = os.path.join(filepath, 'regression')
            filename = 'co2_regression'
    else:
        if classification:
            filepath = os.path.join(filepath, 'classification')
        else:
            filepath = os.path.join(filepath, 'regression')
    return filepath, filename

def load_latest_file(filepath, filename, extension): # for country names or other specificities separated by '_' before the extension
    files = os.listdir(filepath)
    files = [file for file in files if file.endswith(extension)]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(filepath, x)), reverse=True)
    latest_file = files[0]
    return os.path.join(filepath, latest_file)


def separate_target_if_needed(data, separate_Xy=True, classification=False):
    if separate_Xy:        
        if classification:
            X = data.drop(columns=['Co2Grade'])
            y = data['Co2Grade']
            return X, y
        else:
            X = data.drop(columns=['Co2EmissionsWltp'])
            y = data['Co2EmissionsWltp']
            return X, y
    else:
        return data


def load_processed_csv(classification=False, filepath='../data/processed',  chosen_file=None, filename=None, separate_Xy=True):
    filepath, filename = load_processed_helper(classification, filepath, filename)
    filepath = secure_path(filepath)
    if chosen_file is None:
        latest_file = load_latest_file(filepath, filename, '.csv')
    else:
        latest_file = os.path.join(filepath, chosen_file)
    data = pd.read_csv(latest_file)
    return separate_target_if_needed(data, separate_Xy, classification)
    
    

def load_processed_pickle(classification=False, filepath='../data/processed', chosen_file=None, filename=None, separate_Xy=True):
    filepath, filename = load_processed_helper(classification, filepath, filename)
    filepath = secure_path(filepath)
    if chosen_file is None:
        latest_file = load_latest_file(filepath, filename, '.pkl')
    else:
        latest_file = os.path.join(filepath, filename)
    data = pd.read_pickle(latest_file)
    return separate_target_if_needed(data, separate_Xy, classification)
    


########## End of loading processed data ##########




########## General data cleaning ##########

def convert_dtypes(df):
    floats = df.select_dtypes(include=['float']).columns
    df.loc[:, floats] = df.loc[:, floats].astype('float32')

    ints = df.select_dtypes(include=['int']).columns
    df.loc[:, ints] = df.loc[:, ints].astype('float32')

    objs = df.select_dtypes(include=['object']).columns
    df.loc[:, objs] = df.loc[:, objs].astype('category')

    return df


def rename_columns(df, old_names=COL_NAMES_SHORTFORM, new_names=COL_NAMES_LONGFORM):
    name_dict = {}
    if set(old_names).issubset(df.columns): # If either some or all of the short names are in the dataframe...
        name_dict = dict(zip(old_names, new_names)) # Maps concerned short names to long names
    df.rename(name_dict, axis=1, inplace=True)
    return df


def select_countries(df, countries:list):
    return df[df['Country'].isin(countries)]


def drop_irrelevant_columns(df):
    drop_list = [
            'VehicleFamilyIdentification', 'ManufNameMS', 'TypeApprovalNumber', 
            'Type', 'Variant', 'Version', 'VehicleCategory',
           'TotalNewRegistrations', 'Co2EmissionsNedc', 'AxleWidthOther', 'Status',
           'InnovativeEmissionsReduction', 'DeviationFactor', 'VerificationFactor', 'Status',
           'RegistrationYear']
    
    to_drop = [col for col in drop_list if col in df.columns]
    new_df = df.drop(columns=to_drop)
    
    dropped_cols = df.columns.difference(new_df.columns)
    col_names_mapping = dict(zip(COL_NAMES_LONGFORM, COL_NAMES_SHORTFORM))
    _ = [print(f"Irrelevant column dropped: {col} ({col_names_mapping.get(col, col)})") for col in to_drop]
    print() 
    return new_df

def conditional_column_update(df, condition_column, condition_value, target_column, target_value):
    df.loc[df[condition_column] == condition_value, target_column] = target_value
    return df

def clean_manufacturer_columns(df): # VIZ
    df = conditional_column_update(df, 'Make', 'TESLA', 'Pool', 'TESLA') # Pool names cleaning ...
    df = conditional_column_update(df, 'Make', 'HONDA', 'Pool', 'HONDA-GROUP')
    df = conditional_column_update(df, 'Make', 'JAGUAR', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'LAND ROVER', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'RANGE-ROVER', 'Pool', 'TATA-MOTORS')

    df['Make'] = df['Make'].str.upper() # Make names cleaning ...
    df = conditional_column_update(df, 'Make', 'B M W', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'BMW I', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'ROLLS ROYCE I', 'Make', 'ROLLS-ROYCE')
    df = conditional_column_update(df, 'Make', 'FORD (D)', 'Make', 'FORD') 
    df = conditional_column_update(df, 'Make', 'FORD-CNG-TECHNIK', 'Make', 'FORD')
    df = conditional_column_update(df, 'Make', 'FORD - CNG-TECHNIK', 'Make', 'FORD')
    df = conditional_column_update(df, 'Make', 'MERCEDES', 'Make', 'MERCEDES-BENZ')
    df = conditional_column_update(df, 'Make', 'MERCEDES BENZ', 'Make', 'MERCEDES-BENZ') 
    df = conditional_column_update(df, 'Make', 'LYNK & CO', 'Make', 'LYNK&CO')  
    df = conditional_column_update(df, 'Make', 'VOLKSWAGEN VW', 'Make', 'VOLKSWAGEN')
    df = conditional_column_update(df, 'Make', 'VOLKSWAGEN, VW', 'Make', 'VOLKSWAGEN') 
    df = conditional_column_update(df, 'Make', 'VOLKSWAGEN. VW	', 'Make', 'VOLKSWAGEN') 
    df = conditional_column_update(df, 'Make', 'VW', 'Make', 'VOLKSWAGEN')
    return df

def correct_fueltype(df): # VIZ
    df.loc[df['FuelType'] == 'petrol/electric', 'FuelType'] = 'PETROL/ELECTRIC'
    df.loc[df['FuelType'] == 'E85', 'FuelType'] = 'ETHANOL'
    df.loc[df['FuelType'] == 'NG-BIOMETHANE', 'FuelType'] = 'NATURALGAS'
    return df

    # ***** High Level Function: PRE CLEANING ***** #
def data_preprocess(df, countries=None):
    print("Selecting countries...")
    if countries is not None:
        df = select_countries(df, countries)
    print("Converting dtypes (int/float 64 >> 32, object >> category)...")
    df = convert_dtypes(df)
    print("Setting column names to longform...")
    df = rename_columns(df)
    return df
    # ***** End of High Level Function: PRE CLEANING ***** #

    # ***** High Level Function: VIZ CLEANING ***** #
def dataviz_preprocess(df, countries=None):
    df = data_preprocess(df, countries)
    df = drop_irrelevant_columns(df)
    df = clean_manufacturer_columns(df)
    df = correct_fueltype(df)
    df = discretize_co2(df)
    df.loc[df['ElectricRange'].isna(), 'ElectricRange'] = 0 
    df = drop_residual_incomplete_rows(df)  
    df['CommercialName'] = df['CommercialName'].str.replace('[^a-zA-Z0-9\s/-]', '')
    return df
    # ***** End of High Level Function: VIZ ***** #

########## End of General Data Cleaning ##########


########## ML Preprocessing ##########

def remove_columns(df, columns=None, axlewidth=True, engine_capacity=True, fuel_consumption=True):
    columns_to_drop = ['Country', 'ManufacturerName', 'ManufNameOem', 'Type', 'Variant', 
                        'Version', 'Make', 'CommercialName', 'VehicleCategory', 'CommercialName',
                        'TotalNewRegistrations', 'Co2EmissionsNedc', 'WltpTestMass','FuelMode', 
                        'ElectricConsumption', 'InnovativeEmissionsReduction', 'DeviationFactor', 
                        'VerificationFactor', 'Status','RegistrationYear', 'RegistrationDate',
                        'CategoryOf', 'InnovativeEmissionsReductionWltp', 'ID']
    if axlewidth:
        columns_to_drop.append('AxleWidthSteering')
    if engine_capacity:
        columns_to_drop.append('EngineCapacity')
    if fuel_consumption:
        columns_to_drop.append('FuelConsumption')
        
    if columns is None:
        new_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    else:
        new_df = df.drop(columns=[col for col in columns if col in df.columns])
        
    col_names_mapping = dict(zip(COL_NAMES_LONGFORM, COL_NAMES_SHORTFORM))
    dropped_cols = df.columns.difference(new_df.columns)
    _ = [print(f"Column dropped: {col} ({col_names_mapping.get(col, col)})") for col in columns_to_drop]
    print()
    
    return new_df


def remove_fueltype(df, keep_fossil=False):
    rows_t0 = len(df)
    
    if keep_fossil:
        df.drop(df.loc[df['FuelType'].isin(['HYDROGEN', 'ELECTRIC', 'UNKNOWN'])].index, inplace=True)

    else:
        df = df.drop(columns=['FuelType'])
        
    rows_t1 = len(df)
    print(f"FuelType rows dropped:{rows_t0 - rows_t1}")
    
    return df


def standardize_innovtech(df, drop=False):
    if drop:
        df = df.drop(columns=['InnovativeTechnology'])
    else:
        df['InnovativeTechnology'].fillna(0, inplace=True)
        df.loc[df['InnovativeTechnology'] != 0, 'InnovativeTechnology'] = 1
        df['InnovativeTechnology'] = df['InnovativeTechnology'].astype(int)
    return df
    
    
def drop_residual_incomplete_rows(df):
    rows_t0 = len(df)
    for col in df.columns:
        if df[col].isna().mean() <= 0.05:
            df = df[df[col].notna()]

    rows_t1 = len(df)
    print(f"Incomplete rows dropped:{rows_t0 - rows_t1}")
    return df


# ***** High Level Function: ML CLEANING ***** #
def ml_preprocess(df, countries=None, 
                     rem_fuel_consumption=True, 
                     rem_axlewidth=True, 
                     rem_engine_capacity=True,
                     electricrange_nantozero=True):
    
    rows_t0 = len(df)
    
    df_new = data_preprocess(df, countries)
    
    print("Removing redundant, useless, empty columns...")
    df_new = drop_irrelevant_columns(df_new)
    
    print("Removing some other columns...")
    df_new = remove_columns(df_new, axlewidth=rem_axlewidth, 
                            engine_capacity=rem_engine_capacity, 
                            fuel_consumption=rem_fuel_consumption)  
    
    print("binarizing InnovativeTechnology...")
    df_new = standardize_innovtech(df_new)
    
    print("Setting ElectricRange missing values to 0...")
    if electricrange_nantozero:
        df_new.loc[df_new['ElectricRange'].isna(), 'ElectricRange'] = 0
    
    print("Dropping rows with incomplete data if under 5%...")   
    df_new = drop_residual_incomplete_rows(df_new)

    rows_t1 = len(df_new)
    print(f"TOTAL NUMBER OF ROWS DROPPED:{rows_t0 - rows_t1}\n")
    print(f"FINAL NUMBER OF ROWS:{rows_t1}\n")
    print(f"TOTAL NUMBER OF COLUMNS DROPPED:{df.columns.difference(df_new.columns).shape[0]}\n")
    print(f"FINAL NUMBER OF COLUMNS:{len(df_new.columns)}\n")
    
    
    return df_new
# ***** end of High Level Function: ML CLEANING ***** #

########## End of ML Preprocessing ##########



########## Feature Engineering ##########

def discretize_co2(df):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    bins = [-1, 100, 120, 140, 160, 200, 250, 1000]

    grades = pd.cut(df['Co2EmissionsWltp'], bins=bins, labels=labels)
    df['Co2Grade'] = grades
    return df    


def discretize_electricrange(df, to_dummies=False):
    df['ElectricRange'].fillna(0, inplace=True)
    bins = [-float('inf'),0,50,100,150,300]
    labels = ['NO_RANGE', '0to50', '50to100', '100to150', '150+']
    df['ElectricRange'] = pd.cut(df['ElectricRange'], bins=bins, labels=labels)

    if to_dummies:
        df = df.join(pd.get_dummies(data=df['ElectricRange'], dtype=int, prefix='ElecRange'))
        df.drop('ElectricRange', axis=1, inplace=True)

    return df


def get_classification_data(df):
        df = discretize_co2(df)
        df = df.drop(columns=['Co2EmissionsWltp'])
        return df



def dummify(df, column):
    dummies = pd.get_dummies(data=df[column], dtype=int, prefix=f"{column}")
    df = df.join(dummies)
    df.drop(columns=[column], inplace=True)
    return df

def dummify_all_categoricals(df, dummy_columns=None, should_discretize_electricrange=True):
    if dummy_columns is None:
        dummy_columns = ['Pool', 'FuelType']
        if should_discretize_electricrange:
            df = discretize_electricrange(df, to_dummies=True) 
        
    for column in dummy_columns:
        df = dummify(df, column)
        
    return df
    
########## End of Feature Engineering ##########



########## Persistence ##########


def save_processed_data(df, filepath=None, country_names=None, classification=False, pickle=True):
    if filepath is None:
        filepath = '../data/processed'
    if classification:
        filepath = os.path.join(filepath, 'classification')
        filename = 'co2_classification'
    else:
        filepath = os.path.join(filepath, 'regression')
        filename = 'co2_regression'
    
    os.makedirs(filepath, exist_ok=True)

    # append a number to the filename if a file with the same name already exists
    counter = 1
    while os.path.exists(os.path.join(
        filepath, f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}.pkl" if pickle\
            else f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}.csv")):
        counter +=1

    if classification:
        df = get_classification_data(df)

    if pickle:
        df.to_pickle(
            os.path.join(filepath, f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}.pkl"))
    else:
        df.to_csv(
            os.path.join(filepath, f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}.csv"), index=False)

    saved_path = os.path.join(filepath, f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}")
    print(f"Data saved to {saved_path}")




def save_model(model, model_type='other'):
    model_name = type(model).__name__
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = '../models'
    filename = f"{model_name}_{timestamp}"
    os.makedirs(filepath, exist_ok=True)
    
    if model_type == 'xgb' and isinstance(model, (XGBClassifier, XGBRegressor)):
        filename += '.model'
        full_path = os.path.join(filepath, filename)
        model.save_model(full_path)
    elif model_type == 'keras' and isinstance(model, Model):
        filename += '.h5'
        full_path = os.path.join(filepath, filename)
        model.save(full_path)
    else:
        filename += '.pkl'
        full_path = os.path.join(filepath, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
    print(f'Model saved at {full_path}')
    
    
    
    
def save_shap_values(shap_values, shap_sample):
    filename_prefix = type(shap_values).__name__
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = '../output/interpretability'
    filename_shap = f"{filename_prefix}_shap_values_{timestamp}.csv"
    filename_explainer = f"{filename_prefix}_explainer_{timestamp}.csv"
    os.makedirs(filepath, exist_ok=True)
    
    full_path_shap = os.path.join(filepath, filename_shap)
    full_path_explainer = os.path.join(filepath, filename_explainer)
    
    # Save SHAP values to CSV
    shap_values_np = np.concatenate(shap_values, axis=0) if isinstance(shap_values, list) else np.array(shap_values)
    shap_values_df = pd.DataFrame(shap_values_np, columns=shap_sample.columns)
    shap_values_df.to_csv(full_path_shap, index=False)
    print(f'SHAP values saved at {full_path_shap}')

    # Save explainer information to CSV
    with open(full_path_explainer, 'w') as explainer_file:
        explainer_file.write("feature\n")
        explainer_file.write("\n".join(map(str, shap_sample.columns)))
    print(f'Explainer information saved at {full_path_explainer}')