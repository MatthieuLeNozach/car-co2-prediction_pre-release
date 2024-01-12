import os
import pandas as pd
import zipfile
import kaggle




########## Fetching data from Kaggle ##########
def download_co2_data(auth_file_path, filepath='data/raw'):
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

def load_co2_data(dataset_name="automobile-co2-emissions-eu-2021", filepath='data/raw'):
    with zipfile.ZipFile(f'{filepath}/{dataset_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(filepath)
    filename = zip_ref.namelist()[0] 
    data = pd.read_csv(f'{filepath}/{filename}', low_memory=False)
    return data

def download_and_load_co2_data(auth_file_path, filepath='data/raw'):
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
    to_drop = [
            'VehicleFamilyIdentification', 'ManufNameMS', 'TypeApprovalNumber', 
            'Type', 'Variant', 'Version', 'CommercialName', 'VehicleCategory',
           'TotalNewRegistrations', 'Co2EmissionsNedc', 'AxleWidthOther', 'Status',
           'InnovativeEmissionsReduction', 'DeviationFactor', 'VerificationFactor', 'Status',
           'RegistrationYear']
    
    df = df.drop(columns=[col for col in to_drop if col in df.columns])
    return df


def conditional_column_update(df, condition_column, condition_value, target_column, target_value):
    df.loc[df[condition_column] == condition_value, target_column] = target_value
    return df

def clean_manufacturer_columns(df): # VIZ
    df = conditional_column_update(df, 'Make', 'HONDA', 'Pool', 'HONDA-GROUP')
    df = conditional_column_update(df, 'Make', 'JAGUAR', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'LAND ROVER', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'RANGE-ROVER', 'Pool', 'TATA-MOTORS')

    df['Make'] = df['Make'].str.upper()
    df = conditional_column_update(df, 'Make', 'B M W', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'BMW I', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'ROLLS ROYCE I', 'Make', 'ROLLS-ROYCE')
    df = conditional_column_update(df, 'Make', 'FORD (D)', 'Make', 'FORD') 
    df = conditional_column_update(df, 'Make', 'FORD-CNG-TECHNIK', 'Make', 'FORD')
    df = conditional_column_update(df, 'Make', 'FORD - CNG-TECHNIK', 'Make', 'FORD')
    df = conditional_column_update(df, 'Make', 'MERCEDES', 'Make', 'MERCEDES-BENZ')
    df = conditional_column_update(df, 'Make', 'MERCEDES BENZ', 'Make', 'MERCEDES-BENZ') 
    df = conditional_column_update(df, 'Make', 'LYNK & CO', 'Make', 'LYNK&CO')    
 
    return df


def column_remover(df, columns=None):
    columns_to_drop = ['Country', 'Type', 'Variant', 'Version', 'Make', 'CommercialName', 'VehicleCategory',
                        'TotalNewRegistrations', 'Co2EmissionsNedc', 'WltpTestMass','FuelMode', 
                        'ElectricConsumption', 'InnovativeEmissionsReduction', 'DeviationFactor', 
                        'VerificationFactor', 'Status','RegistrationYear', 'RegistrationDate',
                        'Pool', 'CategoryOf']
    if columns is None:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    else:
        df = df.drop(columns=[col for col in columns if col in df.columns])
    return df


def correct_fueltype(df): # VIZ
    df.loc[df['FuelType'] == 'petrol/electric', 'FuelType'] = 'PETROL/ELECTRIC'
    df.loc[df['FuelType'] == 'E85', 'FuelType'] = 'ETHANOL'
    df.loc[df['FuelType'] == 'NG-BIOMETHANE', 'FuelType'] = 'NATURALGAS'
    return df


def remove_fueltype(df, keep_fossil=False):
    if keep_fossil:
        df = df.drop(columns=['FuelType'])
    else:
        df.drop(df.loc[df['FuelType'].isin(['HYDROGEN', 'ELECTRIC', 'UNKNOWN'])].index, inplace=True)
    return df


def standardize_innovtech(df, drop=True):
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
    print(f"Number of rows dropped:{rows_t0 - rows_t1}")
    return df


def electricrange_discretization(df, to_dummies=False):
    df['ElectricRange'].fillna(0, inplace=True)
    bins = [-float('inf'),0,50,100,150,300]
    labels = ['NO_RANGE', '0to50', '50to100', '100to150', '150+']
    df['ElectricRange'] = pd.cut(df['ElectricRange'], bins=bins, labels=labels)

    if to_dummies:
        df = df.join(pd.get_dummies(data=df['ElectricRange'], dtype=int, prefix='ElecRange'))
        df.drop('ElectricRange', axis=1, inplace=True)
    
    return df

def co2_grade_discretization(df):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    bins = [0, 100, 120, 140, 160, 200, 250, 500]

    grades = pd.cut(df['Co2EmissionsWltp'], bins=bins, labels=labels)
    df['Co2Grade'] = grades
    return df    
    


def get_classification_data(df):
        df = co2_grade_discretization(df)
        df = df.drop(columns=['Co2EmissionsWltp'])
        return df



def save_clean_data(df, classification=False, pickle=True, filepath='data/processed'):
    if classification:
        filename = 'co2_classification'
        get_classification_data(df)
    else:
        filename = 'co2_regression'
    if pickle:
        df.to_pickle(f"{filepath}/{filename.split('.')[0]}.pkl")
    df.to_csv(f'{filepath}/{filename}.csv', index=False)
    print(f"Data saved to {filepath}/{filename}")


def ml_preprocessing(df):
    df = column_remover(df)
    df = remove_fueltype(df)
    df = standardize_innovtech(df)
    df = electricrange_discretization(df, to_dummies=True)
    df = drop_residual_incomplete_rows(df)
    
    return df