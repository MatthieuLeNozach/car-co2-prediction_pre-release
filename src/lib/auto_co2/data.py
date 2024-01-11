import os
import pandas as pd
import zipfile
import kaggle




########## Fetching data from Kaggle ##########
def download_co2_data(auth_file_path, filepath='.'):
    dataset_id = 'matthieulenozach/automobile-co2-emissions-eu-2021'
    dataset_name = dataset_id.split('/')[-1]  # Get the dataset name

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
    return dataset_name

def load_co2_data(dataset_name="automobile-co2-emissions-eu-2021", filepath='.'):
    with zipfile.ZipFile(f'{filepath}/{dataset_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(filepath)
    filename = zip_ref.namelist()[0]  # Get the first file name in the zip
    data = pd.read_csv(f'{filepath}/{filename}', low_memory=False)
    return data

def download_and_load_co2_data(auth_file_path, filepath='.'):
    dataset_name = download_co2_data(auth_file_path, filepath)
    data = load_co2_data(dataset_name, filepath)
    return data

########## End of fetching data from Kaggle ##########


########## Preprocessing data ##########
def convert_dtypes(df):
    floats = df.select_dtypes(include=['float']).columns
    df.loc[:, floats] = df.loc[:, floats].astype('float32')

    ints = df.select_dtypes(include=['int']).columns
    df.loc[:, ints] = df.loc[:, ints].astype('float32')

    objs = df.select_dtypes(include=['object']).columns
    df.loc[:, objs] = df.loc[:, objs].astype('category')

    return df


def rename_columns(df):
    abbrev_list = [
        'ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk',
        'Cn', 'Ct', 'Cr', 'r', 'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)',
        'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'IT',
        'Ernedc (g/km)', 'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration',
        'Fuel consumption ', 'Electric range (km)'
        ]

    nom_colonne_list = [
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
    
    name_dict = dict(zip(abbrev_list, nom_colonne_list))
    df.rename(name_dict, axis=1, inplace=True)
    return df

def select_countries(df, countries:list):
    return df[df['Country'].isin(countries)]

########## End of preprocessing data ##########


########## Data cleaning ##########

def drop_irrelevant_columns(df):
    df = df.drop(columns=[
            'VehicleFamilyIdentification', 'ManufNameMS', 'TypeApprovalNumber', 
            'Type', 'Variant', 'Version', 'CommercialName', 'VehicleCategory',
           'TotalNewRegistrations', 'Co2EmissionsNedc', 'AxleWidthOther', 'Status',
           'InnovativeEmissionsReduction', 'DeviationFactor', 'VerificationFactor', 'Status',
           'RegistrationYear'])
    return df


def conditional_column_update(df, condition_column, condition_value, target_column, target_value):
    df.loc[df[condition_column] == condition_value, target_column] = target_value
    return df
    

def clean_manufacturer_columns():
    df = conditional_column_update(df, 'Make', 'HONDA', 'Pool', 'HONDA-GROUP')
    df = conditional_column_update(df, 'Make', 'JAGUAR', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'LAND ROVER', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'RANGE-ROVER', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'B M W', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'BMW I', 'Make', 'BMW')


    
def drop_residual_incomplete_rows(df):
    for col in df.columns:
        if df[col].isna().mean() <= 0.05:
            df = df[df[col].notna()]
    return df