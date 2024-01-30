
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit.components.v1 as components
from PIL import Image
import plotly.express as px


import intro
import dataviz
import model
import demo
import conclusion
import accueil


# Voir si c'est a mettre dans les pages ou la.
st.set_page_config(layout="wide")   # pb eventuel 
st.set_option('deprecation.showPyplotGlobalUse', False)


#----------------------------------------------Chargement des données

# Definition des chemins et noms 
file_path = './data/'     
file_name_no_dummies = "c02_non_dummies_reduit_2e5.pkl"      # pour affichage avant dummies         
file_name_preprocessed = "c02_reduit_2e5.pkl"               # apres Nbook preprocess
           

# Fonction pour charger les données 
@st.cache_data
def load_data(file_name):
    data = pd.read_pickle(file_path + file_name)
    return data


# Chargement des données    
# Session State provides a dictionary-like interface where you can save information that is preserved between script reruns or multipage application
# st.session_state["my_key"] or st.session_state.my_key.

# Chargement df_preprocessed
if 'df' not in st.session_state:            # ou df = load_data()            
    st.session_state['df'] = load_data(file_name = file_name_preprocessed)
df = st.session_state.df

# Chargement df_no_dummies
if 'df_no_dum' not in st.session_state:                    
    st.session_state['df_no_dum'] = load_data(file_name = file_name_no_dummies)
df_no_dum = st.session_state.df_no_dum


#------------------------------------------------SOMMAIRE 



st.sidebar.image("./data/images/pollution3.png", width=250, use_column_width="auto" )

st.sidebar.write("# Emissions CO2 des vehicules")

PAGES = { 
    "Acceuil": accueil,
    "Présentation": intro,
    "Exploration": dataviz,
    "Modèles ML": model,
    "Démonstration": demo,
    "Conclusion, perspectives": conclusion
    }

selection = st.sidebar.radio("Menu", list(PAGES.keys()))
page = PAGES[selection]
page.app(df)



## Affichage des auteurs et mentor en bas de la sidebar:
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write("""
                 **Auteurs :**  
                CALMETTES Ludovic  
                LE NOZACH Matthieu  
                KAPNANG Pierre olivier  
                NOUYETOWA-TOLOFON  Hippolyte                 
                 """)

st.sidebar.write("""
                 **Mentor:**  
                OUERTANI Khalil
                 """)
