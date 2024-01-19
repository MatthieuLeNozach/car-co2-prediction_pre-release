import os
import argparse
import sys
sys.path.insert(0, '../src/')
import importlib
import time

import pandas as pd
import kaggle

import auto_co2 as co2

importlib.reload(co2)



parser = argparse.ArgumentParser()
parser.add_argument('--countries', nargs='+', default=['FR', 'DE']) # nargs='+' means 1 or more arguments
parser.add_argument('--save', type=int, default=3)
args = parser.parse_args()

print(f"\nCountries arg: {args.countries}")
print(f"Save arg: {args.save}")

countries_str = '-'.join(args.countries)
save_tables = args.save in [1, 3]
save_plots = args.save in [2, 3]


print("Loading the data...")
df = co2.data.load_co2_data()
df = co2.data.rename_columns(df)
print("Data successfully loaded!")

# Inspection des données
print("Generating HTML tables on raw data...")
co2.styles.displayer(df, title=f'APERCU DU JEU DE DONNEES: {countries_str}', save=save_tables)
co2.styles.display_info(df, title=f'DONNEES BRUTES: {countries_str}', save=save_tables) 
co2.styles.display_na(df, title=f'PART DE VALEURS MANQUANTES DANS LE JEU DE DONNEES: {countries_str}', save=save_tables)    
time.sleep(1)

print(df.info(), '\n')
# Nettoyage des données
print("Cleaning the data...")
df = co2.data.ml_preprocess(df, countries=args.countries,
                               rem_axlewidth=True,
                               rem_fuel_consumption=True,
                               rem_engine_capacity=True,
                               electricrange_nantozero=True)


# Inspection des features
print("Generating HTML tables on processed data...")
co2.styles.display_info(df, title=f'DONNEES NETTOYEES: {countries_str}', save=save_tables) 
co2.styles.display_describe(df, title=f'STATISTIQUES DESCRIPTIVES: {countries_str}', save=save_tables)
time.sleep(1)

print("Generating correlation heatmap...")
co2.viz.plot_correlation_heatmap(df, interactive=False, save=save_plots, title=f': {countries_str}')
print("Generating feature distributions plot...")
co2.viz.plot_feature_distributions(df, interactive=False, save=save_plots, title=f': {countries_str}')
print("Generating QQ plot...")
co2.viz.plot_qqplots(df, interactive=False, save=save_plots, title=f': {countries_str}')
time.sleep(1)

# One-hot encoding des variables catégorielles
print("One-hot encoding categorical variables...")
df = co2.data.dummify_all_categoricals(df, should_discretize_electricrange=True)

print('\n', df.info(), '\n')
time.sleep(1)
# Sauvegarde du jeu de données prêt à l'emploi pour la régression
print("Saving the processed data...")
co2.data.save_processed_data(df, classification=True, country_names=countries_str, pickle=False)


print("\nDataset successully preprocessed and saved!")
print("Plots saved in output/plots, tables saved in output/tables/, data saved in output/data/processed")