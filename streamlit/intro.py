import streamlit as st
import pandas as pd
from PIL import Image   # Utile ?


# --------------------------------Page "Introduction"

def app(df) : 
    
    st.title("Datascience et émissions CO2 des voitures")
    st.markdown("---")  

    # Présentation du projet DS CO2
    st.write("""
             ### Objectifs:  

             - Construire un modèle d'apprentissage automatique fiable pour prédire les émissions de CO2 dans différents types de voitures, en fonction de leurs caractéristiques (Puissance, masse, carburant..).
             - Fournir des outils d'analyse sur l'influence de ces caractéristiques, et sur leur répartition dans le marché automobile.  
             - Développer des connaissances dans ce domaine pour accompagner les décideurs privés et publics.

             Le Transport est responsables de 29 % des émissions de CO2 dans le monde.
             """)
   
    ## Enjeux du sujet (contexte)
    # st.write(<center> "https://thinkr.fr/wp-content/uploads/my_image.PNG" </center>)
   


    st.write("""
        ### Données source:  
        Source du jeu de données: [European Environment Agency](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b)  
        C'est une structure de l'Union Europeenne qui enregistre toutes les nouvelles immatriculations de voitures en europe.  
             
        La taille du ce jeu de données est très importante: 9 millions de véhicules pour la seule année 2021.  
        - Dataset d'exploration et analyse des données: Europe 2021        
        Entrainement des modèles sur un échatillon réduit, pour garder un temps de traitement raisonnable.  
        - Dataset d'entrainement des modèles: France & Allemagne 2021  
        - Il comporte **4 500 000 enregistrements sur 38 variables**.   
    
        ### Méthodologie
        1. Collecte et nettoyage des données  
        2. Analyse exploratoire  
        3. **Developpement de modèles prédictifs**  
        4. Analyse des résultats et modèles
        5. Prolongements
             """)

    st.write("""
                ### Regression ou Classification ?
                **Variable cible** : Emission de CO2 des voitures, en g/km  
                C'est un problème de regression, mais pour des raisons pédagogiques, nous avons souhaité explorer deux approches :               

                 """)
    col1, col2 = st.columns([0.7,0.3], gap = 'small')
    with col1:   
        st.write("""
                - **Problème de Regression**: On cherche à prédire la valeur des émisssions de CO2. 
                - **Problème de Classification**: On cherche à prédire la classe d'apartenance des émisssions de CO2.  
                Les normes Européennes définissent un classement pour le niveau de CO2 emis, et on peut donc imaginer que cette approche soit à privilégier.  
                """)

    with col2:
            classes = "./data/images/classes.png"
            st.image(classes, use_column_width = "auto", output_format = "PNG")


