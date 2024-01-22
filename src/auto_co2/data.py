import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
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
        'Fuel consumption ', 'Electric range (km)']

COL_NAMES_LONGFORM = [ # Official column names from EEA's' website
        'ID', 'Country', 'VehicleFamilyIdentification', 'Pool', 'ManufacturerName', 'ManufNameOem',
        'ManufNameMS', 'TypeApprovalNumber', 'Type', 'Variant', 'Version', 'Make', 'CommercialName',
        'VehicleCategory', 'CategoryOf', 'TotalNewRegistrations', 'MassRunningOrder',
        'WltpTestMass', 'Co2EmissionsNedc', 'Co2EmissionsWltp',
        'BaseWheel', 'AxleWidthSteering', 'AxleWidthOther', 'FuelType', 'FuelMode',
        'EngineCapacity', 'EnginePower', 'ElectricConsumption',
        'InnovativeTechnology', 'InnovativeEmissionsReduction',
        'InnovativeEmissionsReductionWltp', 'DeviationFactor', 'VerificationFactor',
        'Status', 'RegistrationYear', 'RegistrationDate', 'FuelConsumption', 'ElectricRange']

TOTALLY_UNUSABLE_COLUMNS = [
            'VehicleFamilyIdentification', 'ManufNameMS', 'TypeApprovalNumber', 
            'Type', 'Variant', 'Version', 'VehicleCategory',
           'TotalNewRegistrations', 'Co2EmissionsNedc', 'AxleWidthOther', 'Status',
           'InnovativeEmissionsReduction', 'DeviationFactor', 'VerificationFactor', 'Status',
           'RegistrationYear']

VIZ_COLUMNS_SELECTION = ['Make', 'RegistrationDate', 'CommercialName', 'MassRunningOrder', 'Co2EmissionsWltp', 'BaseWheel', 'EnginePower', 
                        'InnovativeTechnology', 'ElectricRange', 'Pool', 'FuelType', 'FuelConsumption', 'Country', 'AxleWidthSteering', 'ID']

ML_COLUMNS_SELECTION = ['MassRunningOrder', 'Co2EmissionsWltp', 'BaseWheel', 'EnginePower', 
                        'InnovativeTechnology', 'ElectricRange', 'Pool', 'FuelType']



########## End of Constants ##########



########## Security tools ##########

def secure_path(filepath:str):
    root = Path('../')
    absolute_filepath = Path(filepath).resolve()
    
    if root not in absolute_filepath.parents:
        raise ValueError(f"Path {filepath} is not within the repository root, please save the file within the repository boundaries and try again.")


########## End of security tools ##########




########## Fetching raw data from Github ##########

def download_file(url:str, filepath:str):
    """
    Downloads a file from a URL.

    Parameters:
    url (str): The URL of the file to download.
    filepath (str): The local path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"File has been downloaded to {filepath}.")

def unzip(zipfile_path, target_path):
    """
    Extracts a ZIP file.

    Parameters:
    zipfile_path (str): The path of the ZIP file.
    extract_path (str): The path where the ZIP file will be extracted.
    """
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)

def download_and_load_co2_data(filepath='../data/raw'):
    """
    Downloads and loads CO2 data.

    Parameters:
    filepath (str): The directory where the data file will be saved and loaded from.

    Returns:
    DataFrame: The loaded data.
    """
    dataset_name = "automobile-co2-emissions-eu-2021"
    url = 'https://github.com/MatthieuLeNozach/car-co2-prediction_pre-release/blob/0.2.01-alpha/data/raw/automobile-co2-emissions-eu-2021.zip'
    zipfile_path = os.path.join(filepath, f"{dataset_name}.zip")
    csv_file_path = os.path.join(filepath, "auto_co2_eur_21_raw.csv")

    if not os.path.isfile(csv_file_path):
        if not os.path.isfile(zipfile_path):
            download_file(url, zipfile_path)
        unzip(zipfile_path, filepath)

    data = pd.read_csv(csv_file_path, low_memory=False)
    return data

########## End of fetching raw data from Github  ##########




########## General data cleaning ##########

def convert_dtypes(df):
    """
    Converts the data types of a DataFrame to lighter dtypes (int/float 64 to 32, object to category).

    Parameters:
    df (DataFrame): The DataFrame to convert.

    Returns:
    DataFrame: The converted DataFrame.
    """
    floats = df.select_dtypes(include=['float']).columns
    df.loc[:, floats] = df.loc[:, floats].astype('float32')

    ints = df.select_dtypes(include=['int']).columns
    df.loc[:, ints] = df.loc[:, ints].astype('float32')

    objs = df.select_dtypes(include=['object']).columns
    df.loc[:, objs] = df.loc[:, objs].astype('category')

    return df



def rename_columns(df, old_names=COL_NAMES_SHORTFORM, new_names=COL_NAMES_LONGFORM):
    """
    DataFrame columns soft renamer, wont raise an error some old names are missing.

    Parameters:
    df (DataFrame): The DataFrame to rename.
    old_names (list): The old column names.
    new_names (list): The new column names.

    Returns:
    DataFrame: The DataFrame with renamed columns.
    """
    name_dict = {}
    if set(old_names).issubset(df.columns): # If either some or all of the short names are in the dataframe...
        name_dict = dict(zip(old_names, new_names)) # Maps concerned short names to long names
    df.rename(name_dict, axis=1, inplace=True)
    return df


def select_countries(df, countries:list):
    """
    Filters a DataFrame based on a list of countries.

    Parameters:
    df (DataFrame): The DataFrame to filter.
    countries (list): The list of countries to filter by.

    Returns:
    DataFrame: The filtered DataFrame.
    """
    return df[df['Country'].isin(countries)]


def soft_drop_columns(df, columns):
    """
    Attempts to drop specified columns from a DataFrame without raising an error if some columns are not present.

    Parameters:
    df (DataFrame): The DataFrame to remove columns from.
    columns (list): The list of columns to remove.

    Returns:
    DataFrame: The DataFrame with removed columns.
    """
    # Only keep columns that are in the DataFrame
    columns_to_drop = [col for col in columns if col in df.columns]

    return df.drop(columns=columns_to_drop)

def conditional_column_update(df, condition_column, condition_value, target_column, target_value):
    """
    Updates a target column in a based on a condition in any other column (EXACT STRING MATCH).
    """
    df.loc[df[condition_column] == condition_value, target_column] = target_value
    return df


def soft_conditional_column_update(df, condition_column, condition_value, target_column, target_value):
    """
    Updates a target column in a DataFrame based on a condition in any other column (PARTIAL STRING MATCH).
    """
    df.loc[df[condition_column].str.contains(condition_value, na=False), target_column] = target_value
    return df

def clean_manufacturer_columns(df): # VIZ
    """
    Cleans and standardizes the 'Make' and 'Pool' columns in the DataFrame.
    """
    
    df['Make'] = df['Make'].str.upper() 
    df['Pool'] = df['Pool'].str.upper() 
    df = soft_conditional_column_update(df, 'Make', 'TESLA', 'Pool', 'TESLA') # Pool names cleaning ...
    df = conditional_column_update(df, 'Make', 'HONDA', 'Pool', 'HONDA-GROUP')
    df = conditional_column_update(df, 'Make', 'JAGUAR', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'LAND ROVER', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'RANGE-ROVER', 'Pool', 'TATA-MOTORS')
    df = soft_conditional_column_update(df, 'Pool', 'TESLA', 'Make', 'TESLA') # Pool names cleaning ...

    # Make names cleaning ...
    df = conditional_column_update(df, 'Make', 'B M W', 'Make', 'BMW')
    df = soft_conditional_column_update(df, 'Make', 'BMW', 'Make', 'BMW')
    df = soft_conditional_column_update(df, 'Make', 'ROYCE', 'Make', 'ROLLS-ROYCE')
    df = soft_conditional_column_update(df, 'Make', 'FORD', 'Make', 'FORD') 
    df = soft_conditional_column_update(df, 'Make', 'MERCEDES', 'Make', 'MERCEDES-BENZ')
    df = soft_conditional_column_update(df, 'Make', 'LYNK', 'Make', 'LYNK&CO')  
    df = soft_conditional_column_update(df, 'Make', 'VOLKSWAGEN', 'Make', 'VOLKSWAGEN')
    df = soft_conditional_column_update(df, 'Make', 'VW', 'Make', 'VOLKSWAGEN')
    return df



def clean_carname_columns(df):
    df['CommercialName'] = df['CommercialName'].str.upper()
    df['CommercialName'] = df['CommercialName'].str.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    df = soft_conditional_column_update(df, 'CommercialName', 'RAV4', 'CommercialName', 'RAV4')

    return df

def correct_fueltype(df):
    """
    Corrects the 'FuelType' column in the DataFrame.

    'petrol/electric' is corrected to 'PETROL/ELECTRIC'
    'E85' is corrected to 'ETHANOL'
    'NG-BIOMETHANE' is corrected to 'NATURALGAS'

    Parameters:
    df (DataFrame): The DataFrame to correct 'FuelType' in.

    Returns:
    DataFrame: The DataFrame with the corrected 'FuelType' column.
    """
    fueltype_mapping = {'petrol/electric': 'PETROL/ELECTRIC', 'E85': 'ETHANOL', 'NG-BIOMETHANE': 'NATURALGAS'}
    df['FuelType'] = df['FuelType'].replace(fueltype_mapping)
    return df


    # ***** High Level Function: PRE CLEANING ***** #
    
def data_preprocess(df, countries=None):
    """
    Performs initial preprocessing on the DataFrame.

    This function selects specific countries, converts data types, and renames columns.

    Parameters:
    df (DataFrame): The DataFrame to preprocess.
    countries (list, optional): A list of countries to select. If None, all countries are selected. Defaults to None.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    print("Selecting countries...")
    if countries is not None:
        df = select_countries(df, countries)
    print("Converting dtypes (int/float 64 >> 32, object >> category)...")
    df = convert_dtypes(df)
    print("Setting column names to longform...")
    df = rename_columns(df)
    return df

    # ***** End of High Level Function: PRE CLEANING ***** #


########## End of General Data Cleaning ##########


########## ML Preprocessing ##########

def keep_only_selected_columns(df, columns_to_keep):
    """
    Keeps only specified columns in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to keep columns from.
    columns_to_keep (list): The list of columns to keep.

    Returns:
    DataFrame: The DataFrame with only the specified columns.
    """
    return df.reindex(columns=columns_to_keep)


def print_dropped_columns(df, columns):
    """
    Prints the names of the columns that have been dropped from the DataFrame.
    """
    col_names_mapping = dict(zip(COL_NAMES_LONGFORM, COL_NAMES_SHORTFORM))
    _ = [print(f"Column dropped: {col} ({col_names_mapping.get(col, col)})") for col in columns if col in df.columns]
    print()



def remove_non_fossil_fuels(df):
    """
    Removes rows from the DataFrame where the vehicle is not fossil fuelled.

    Rows where 'FuelType' is 'HYDROGEN', 'ELECTRIC', or 'UNKNOWN' are dropped.

    Parameters:
    df (DataFrame): The DataFrame to remove non-fossil fuel types from.

    Returns:
    DataFrame: The DataFrame with non-fossil fuel types removed.
    """
    initial_row_count = len(df)
    df.drop(df.loc[df['FuelType'].isin(['HYDROGEN', 'ELECTRIC', 'UNKNOWN'])].index, inplace=True)
    final_row_count = len(df)
    print(f"Non-fossil FuelType rows dropped: {initial_row_count - final_row_count}")
    return df

def drop_innovtech(df):
    """
    Drops the 'InnovativeTechnology' column from the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to drop 'InnovativeTechnology' from.

    Returns:
    DataFrame: The DataFrame without the 'InnovativeTechnology' column.
    """
    return df.drop(columns=['InnovativeTechnology'])

def standardize_innovtech(df):
    """
    Cleans the 'InnovativeTechnology' column in the DataFrame.
    NaN values in the 'InnovativeTechnology' column are replaced with 0, 
    and all non-zero values have some kind of innovation technology.
    They ar replaced with 1

    Returns:
    DataFrame: The DataFrame with the standardized 'InnovativeTechnology' column.
    """
    df['InnovativeTechnology'].fillna(0, inplace=True)
    df.loc[df['InnovativeTechnology'] != 0, 'InnovativeTechnology'] = 1
    df['InnovativeTechnology'] = df['InnovativeTechnology'].astype(int)
    return df


def discretize_electricrange(df, to_dummies=False):
    """
    Discretizes the 'ElectricRange' column into ranges.

    Parameters:
    df (DataFrame): The DataFrame to process.
    to_dummies (bool, optional): Whether to convert the 'ElectricRange' column to dummy variables. Defaults to False.

    Returns:
    DataFrame: The processed DataFrame.
    """
    df['ElectricRange'].fillna(0, inplace=True)
    bins = [-float('inf'),0,50,100,150,300]
    labels = ['NO_RANGE', '0to50', '50to100', '100to150', '150+']
    df['ElectricRange'] = pd.cut(df['ElectricRange'], bins=bins, labels=labels)

    if to_dummies:
        df = df.join(pd.get_dummies(data=df['ElectricRange'], dtype=int, prefix='ElecRange'))
        df.drop('ElectricRange', axis=1, inplace=True)

    return df

    
def drop_residual_incomplete_rows(df):
    """
    Drops rows with at least 1 missing value, only for columns where the proportion of missing values is less than or equal to 5%.
    This function is intended to be called as last treatment in the preprocessing pipeline to save the maximum amount of data.

    Parameters:
    df (DataFrame): The DataFrame to drop rows from.

    Returns:
    DataFrame: The DataFrame with residual incomplete rows dropped.
    """
    rows_t0 = len(df)
    for col in df.columns:
        if df[col].isna().mean() <= 0.05:
            df = df[df[col].notna()]

    rows_t1 = len(df)
    print(f"Incomplete rows dropped:{rows_t0 - rows_t1}")
    return df


    # ***** High Level Function: VIZ CLEANING ***** #
def dataviz_preprocess(df, countries=None):
    """
    Performs a preprocessing sequence on the DataFrame for data visualization.

    This function applies several preprocessing steps including initial preprocessing, dropping unusable columns, 
    cleaning manufacturer columns, correcting fuel types, discretizing CO2, filling missing electric range values, 
    dropping residual incomplete rows, and cleaning the 'CommercialName' column.

    Parameters:
    df (DataFrame): The DataFrame to preprocess.
    countries (list, optional): A list of countries to select. If None, all countries are selected. Defaults to None.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    rows_t0 = len(df)
    df_new = data_preprocess(df, countries)
    df_new.loc[df_new['ElectricRange'].isna(), 'ElectricRange'] = 0
    df_new.loc[df_new['FuelConsumption'].isna(), 'FuelConsumption'] = 0
    print(df_new.columns)
    df_new = keep_only_selected_columns(df = df_new, columns_to_keep = VIZ_COLUMNS_SELECTION)
    print(f"Number of Tesla entries after soft_drop_columns: {len(df_new[df_new['Pool'] == 'TESLA'])}")
    df_new = clean_manufacturer_columns(df_new)
    print(f"Number of Tesla entries after clean_manufacturer_columns: {len(df_new[df_new['Pool'] == 'TESLA'])}")
    df_new = clean_carname_columns(df_new)
    print(f"Number of Tesla entries before : {len(df_new[df_new['Pool'] == 'TESLA'])}")

    df_new = correct_fueltype(df_new)
    print(f"Number of Tesla entries before : {len(df_new[df_new['Pool'] == 'TESLA'])}")

    df_new = discretize_co2(df_new)
    print(f"Number of Tesla entries before : {len(df_new[df_new['Pool'] == 'TESLA'])}")

    print(df_new['Country'].value_counts())
    df_new = drop_residual_incomplete_rows(df_new) 
    print(f"Number of Tesla entries before : {len(df_new[df_new['Pool'] == 'TESLA'])}")
 
    df_new['CommercialName'] = df_new['CommercialName'].str.replace('[^a-zA-Z0-9\s/-]', '')
    print(f"Number of Tesla entries before : {len(df_new[df_new['Pool'] == 'TESLA'])}")
    

    print(f"TOTAL NUMBER OF COLUMNS DROPPED:{df.columns.difference(df_new.columns).shape[0]}\n")
    print(f"FINAL NUMBER OF COLUMNS:{len(df_new.columns)}\n")
    return df_new
    # ***** End of High Level Function: VIZ ***** #



# ***** High Level Function: ML CLEANING ***** #
def ml_preprocess(df, countries=None, 
                     rem_fuel_consumption=True, 
                     electricrange_nantozero=True,
                     discretize_electricrange=True):
    
    """
    Performs machine learning preprocesing.

    This function applies several preprocessing steps including initial preprocessing, keeping only selected columns, 
    standardizing the 'InnovativeTechnology' column, optionally removing the 'FuelConsumption' column, 
    setting missing 'ElectricRange' values to 0, optionally discretizing the 'ElectricRange' column, 
    and dropping rows with incomplete data.

    Parameters:
    df (DataFrame): The DataFrame to preprocess.
    countries (list, optional): A list of countries to select. If None, all countries are selected. Defaults to None.
    rem_fuel_consumption (bool, optional): Whether to remove the 'FuelConsumption' column. Defaults to True.
    electricrange_nantozero (bool, optional): Whether to set missing 'ElectricRange' values to 0. Defaults to True.
    discretize_electricrange (bool, optional): Whether to discretize the 'ElectricRange' column. Defaults to True.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    
    rows_t0 = len(df)
        
    print("Keeping only selected columns...")
    df_new = keep_only_selected_columns(df = df_new, columns_to_keep = ML_COLUMNS_SELECTION)
    
    print("binarizing InnovativeTechnology...")
    df_new = standardize_innovtech(df_new)
    
    if rem_fuel_consumption:
        print("Removing FuelConsumption column...")
        df_new = df_new.drop(columns=['FuelConsumption'])  
    
    print("Setting ElectricRange missing values to 0...")
    if electricrange_nantozero:
        df_new.loc[df_new['ElectricRange'].isna(), 'ElectricRange'] = 0
        
    if discretize_electricrange:
        print("Discretizing ElectricRange...")
        df_new = discretize_electricrange(df_new, to_dummies=True)
        
    
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
    """
    Discretizes the 'Co2EmissionsWltp' column into grades and adds it as 'Co2Grade' column.

    Parameters:
    df (DataFrame): The DataFrame to process.

    Returns:
    DataFrame: The processed DataFrame.
    """
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    bins = [-1, 100, 120, 140, 160, 200, 250, 1000]

    grades = pd.cut(df['Co2EmissionsWltp'], bins=bins, labels=labels)
    df['Co2Grade'] = grades
    return df    



def get_classification_data(df):
        """
        Prepares the DataFrame for classification by discretizing the 'Co2EmissionsWltp' column and dropping it.

        Parameters:
        df (DataFrame): The DataFrame to process.

        Returns:
        DataFrame: The processed DataFrame.
        """
        df = discretize_co2(df)
        df = df.drop(columns=['Co2EmissionsWltp'])
        return df


def dummify(df, column):
    """
    Converts a categorical column into dummy variables.

    Parameters:
    df (DataFrame): The DataFrame to process.
    column (str): The column to convert.

    Returns:
    DataFrame: The processed DataFrame.
    """
    dummies = pd.get_dummies(data=df[column], dtype=int, prefix=f"{column}")
    df = df.join(dummies)
    df.drop(columns=[column], inplace=True)
    return df


def dummify_all_categoricals(df, dummy_columns=None, max_columns=200):
    """
    Converts all categorical columns into dummy variables, 
    max_columns acts as a safety net to avoid creating too many columns.

    Parameters:
    df (DataFrame): The DataFrame to process.
    dummy_columns (list, optional): The columns to convert. If None, converts all categorical columns. Defaults to None.
    max_columns (int, optional): The maximum number of columns allowed after dummification. Defaults to 200.

    Returns:
    DataFrame: The processed DataFrame.

    Raises:
    ValueError: If the number of potential columns exceeds max_columns.
    """
    if dummy_columns is None:
        dummy_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    potential_new_columns = sum([len(df[column].unique()) for column in dummy_columns])

    if potential_new_columns > max_columns:
        raise ValueError(f"Dummification would result in {potential_new_columns} columns, which exceeds the limit of {max_columns}.")

    for column in dummy_columns:
        df = dummify(df, column)

    return df
    
########## End of Feature Engineering ##########



########## Persistence ##########


def prepare_save_path(filepath=None):
    """
    Prepares the save path for the processed dataset.

    Parameters:
    filepath (str, optional): Defaults to '../data/processed'.

    Returns:
    str: The save path.
    """
    if filepath is None:
        filepath = '../data/processed'
    os.makedirs(filepath, exist_ok=True)
    return filepath

def generate_unique_filename(filepath, filename, country_names=None, pickle=True):
    """
    Generates a unique filename for the processed dataset.

    Parameters:
    filepath (str): The save path.
    filename (str): The base filename.
    country_names (str, optional): Defaults to None.
    pickle (bool, optional): Defaults to True.

    Returns:
    str: The unique filename.
    """
    counter = 1
    while os.path.exists(os.path.join(
        filepath, f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}.pkl" if pickle\
            else f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}.csv")):
        counter +=1
    return f"{filename}_{country_names if country_names else ''}{counter if counter > 1 else ''}"

def save_data(df, filepath=None, country_names=None, filename='co2_data', pickle=True):
    """
    Saves the processed dataset.

    Parameters:
    df (DataFrame): The DataFrame to save.
    filepath (str, optional): Defaults to '../data/processed'.
    country_names (str, optional): Defaults to None.
    filename (str, optional): Defaults to 'co2_data'.
    pickle (bool, optional): Defaults to True.

    Returns:
    None
    """
    filepath = prepare_save_path(filepath)
    unique_filename = generate_unique_filename(filepath, filename, country_names, pickle)

    if pickle:
        df.to_pickle(os.path.join(filepath, f"{unique_filename}.pkl"))
    else:
        df.to_csv(os.path.join(filepath, f"{unique_filename}.csv"), index=False)

    saved_path = os.path.join(filepath, unique_filename)
    print(f"Data saved to {saved_path}")


def save_model(model, model_type='other'):
    """
    Saves the given sklearn/xgboost/keras model to a file.

    Parameters:
    model: The model to save.
    model_type (str, optional): The type of the model. Defaults to 'other'.

    Returns:
    None
    """
    model_name = type(model).__name__
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = '../models'
    filename = f"{model_name}_{timestamp}"
    os.makedirs(filepath, exist_ok=True)
    
    if model_type == 'xgb' and isinstance(model, (XGBClassifier, XGBRegressor)):
        filename += '.model'
    elif model_type == 'keras' and isinstance(model, Model):
        filename += '.h5'
    else:
        filename += '.pkl'

    full_path = os.path.join(filepath, filename)
    
    if model_type in ['xgb', 'keras']:
        model.save(full_path)
    else:
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f'Model saved at {full_path}')

def save_shap_values(shap_values, shap_sample):
    """
    Saves the SHAP values and explainer information to CSV files.

    Parameters:
    shap_values: The SHAP values to save.
    shap_sample: The sample data used for SHAP.

    Returns:
    None
    """
    filename_prefix = type(shap_values).__name__
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = '../output/interpretability'
    os.makedirs(filepath, exist_ok=True)
    
    filename_shap = f"{filename_prefix}_shap_values_{timestamp}.csv"
    filename_explainer = f"{filename_prefix}_explainer_{timestamp}.csv"
    
    full_path_shap = os.path.join(filepath, filename_shap)
    full_path_explainer = os.path.join(filepath, filename_explainer)
    
    shap_values_np = np.concatenate(shap_values, axis=0) if isinstance(shap_values, list) else np.array(shap_values)
    pd.DataFrame(shap_values_np, columns=shap_sample.columns).to_csv(full_path_shap, index=False)
    print(f'SHAP values saved at {full_path_shap}')

    with open(full_path_explainer, 'w') as explainer_file:
        explainer_file.write("feature\n")
        explainer_file.write("\n".join(map(str, shap_sample.columns)))
    print(f'Explainer information saved at {full_path_explainer}')
    
    
    
def load_processed_helper(classification=False, filepath='../data/processed', filename=None):
    """
    Helper function to load processed data.
    Args:
        classification (bool, optional): Whether to load classification data. Defaults to False.
        filepath (str, optional): The path to the processed data directory. Defaults to '../data/processed'.
        filename (str, optional): The name of the file to load. If None, a default filename will be used based on the classification parameter. Defaults to None.

    Returns:
        tuple: A tuple containing the filepath and filename of the loaded data.
    """
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
    """
    Automated feature/target split routine when separate_Xy is True
    Target is Co2Grade if classification is True, else Co2EmissionsWltp
    Returns:
    If `separate_Xy` is True:
        X (DataFrame): The features.
        y (Series): The target.
    If `separate_Xy` is False:
        data (DataFrame): The input data.
    """
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
    """
    Loads a processed CSV file and optionally separates features and target.

    Parameters:
    classification (bool): Whether the task is classification or regression.
    filepath (str): The directory of the file.
    chosen_file (str): The specific file to load. If None, the latest file is loaded.
    filename (str): The base name of the file.
    separate_Xy (bool): Whether to separate the data into features and target.

    Returns:
    DataFrame or tuple: The loaded data, or a tuple of features and target.
    """
    filepath, filename = load_processed_helper(classification, filepath, filename)
    if chosen_file is None:
        latest_file = load_latest_file(filepath, filename, '.csv')

    data = pd.read_csv(latest_file)
    return separate_target_if_needed(data, separate_Xy, classification)
    
    

def load_processed_pickle(classification=False, filepath='../data/processed', chosen_file=None, filename=None, separate_Xy=True):
    """
    Loads a processed pickle file and optionally separates features and target.

    Parameters:
    classification (bool): Whether the task is classification or regression.
    filepath (str): The directory of the file.
    chosen_file (str): The specific file to load. If None, the latest file is loaded.
    filename (str): The base name of the file.
    separate_Xy (bool): Whether to separate the data into features and target.

    Returns:
    DataFrame or tuple: The loaded data, or a tuple of features and target.
    """
    filepath, filename = load_processed_helper(classification, filepath, filename)
    filepath = secure_path(filepath)
    if chosen_file is None:
        latest_file = load_latest_file(filepath, filename, '.pkl')
    else:
        latest_file = os.path.join(filepath, filename)
    data = pd.read_pickle(latest_file)
    return separate_target_if_needed(data, separate_Xy, classification)
    