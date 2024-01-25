import os
import argparse
import sys
sys.path.insert(0, '../src/')
import importlib
import time

import pandas as pd

import auto_co2 as co2

importlib.reload(co2)



parser = argparse.ArgumentParser()
parser.add_argument('--countries', nargs='+', default=['FR', 'DE']) # nargs='+' means 1 or more arguments
parser.add_argument('--save', type=int, default=3)
args = parser.parse_args()

print(f"\nCountries arg: {args.countries}")
print(f"Save arg: {args.save}")

countries_str = '-'.join(args.countries)
SAVE_TABLES = args.save in [1, 3]
SAVE_PLOTS = args.save in [2, 3]
CLASSIFICATION = True


print("Loading the data...")
zip_file_path = '../data/raw/automobile-co2-emissions-eu-2021.zip'
csv_target_path = '../data/raw'


unzipped_file = co2.data.unzip(zipfile_path=zip_file_path, target_path=csv_target_path)
df = pd.read_csv(unzipped_file, low_memory=False)
df = co2.data.data_preprocess(df, countries=args.countries)
print("Data successfully loaded!")

if SAVE_TABLES:
    # Inspection des données
    print("Generating HTML tables on raw data...")
    co2.styles.displayer(df, title=f'APERCU DU JEU DE DONNEES: {countries_str}', save=SAVE_TABLES)
    co2.styles.display_info(df, title=f'DONNEES BRUTES: {countries_str}', save=SAVE_TABLES) 
    co2.styles.display_na(df, title=f'PART DE VALEURS MANQUANTES DANS LE JEU DE DONNEES: {countries_str}', save=SAVE_TABLES)    
    time.sleep(1)
    


print(df.info(), '\n')

# Nettoyage des données
print("Cleaning the data...")
df = co2.data.ml_preprocess(df, countries=args.countries,
                               electricrange_nantozero=True,
                               discretize_electricrange_flag=True)
if SAVE_PLOTS:
    print("Generating correlation heatmap...")
    co2.viz.plot_correlation_heatmap(df, interactive=False, save=SAVE_PLOTS, title=f': {countries_str}')


if SAVE_TABLES:
    # Inspection des features
    print("Generating HTML tables on processed data...")
    co2.styles.display_info(df, title=f'DONNEES NETTOYEES: {countries_str}', save=SAVE_TABLES) 
    co2.styles.display_describe(df, title=f'STATISTIQUES DESCRIPTIVES: {countries_str}', save=SAVE_TABLES)
    print("Tables saved in output/tables/")
    time.sleep(1)


if SAVE_PLOTS:
    print("Generating feature distributions plot...")
    co2.viz.plot_feature_distributions(df, interactive=False, save=SAVE_PLOTS, title=f': {countries_str}')
    print("Generating QQ plot...")
    co2.viz.plot_qqplots(df, interactive=False, save=SAVE_PLOTS, title=f': {countries_str}')
    print("Plots saved in output/plots/")
    time.sleep(1)

# One-hot encoding des variables catégorielles
print("One-hot encoding categorical variables...")
df = co2.data.dummify_all_categoricals(df)
print("Generating Co2 score..")
df = co2.data.get_classification_data(df)


print('\n', df.info(), '\n')
time.sleep(1)
# Sauvegarde du jeu de données prêt à l'emploi pour la régression
print("Saving the processed data...")
co2.data.save_processed_data(df, classification=CLASSIFICATION, country_names=countries_str, pickle=False)


print("\nDataset successully preprocessed and saved!")
print("Processed data file saved in output/data/processed")