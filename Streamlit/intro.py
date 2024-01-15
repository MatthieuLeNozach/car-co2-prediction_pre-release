import streamlit as st
import pandas as pd


# Page "Introduction"
def app(df) : 


    ### Exemples de titre
    st.markdown("<h1 style='position: relative; top: 50%; left: 35%; transform: translate(-43%, -30%); color: red;'>Analysis of CO2 emissions</h1>", unsafe_allow_html=True)   
    st.header("voici le header")
    st.write("### Introduction markdown titre 3")



    st.write('CALMETTES Ludovic, les autres noms')


    # Présentation du projet DS CO2
    st.write('Objectif : Prédire les émissions de CO2 (g/km) de véhicules')

    ## Enjeux du sujet
    st.write('Développer des connaissances dans ce domaine pour accompagner les décideurs privés et publics')
    st.write('le Transport est responsables de 25% des émissions de CO2 dans le monde')

    ## Base de données
    st.write("Source du jeu de données: European Environment Agency (EEA)\n     https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b")


  
    
    ### Les données sources
    # todo: voir s'il faut utiliser les données non transformées (avec nom du modele par exemple), ou changer le titre "apres preprocessing"
    # TODO: reordonner le df 
    st.write("### Les données")
    st.write("Taille des données", df.shape)    # todo: nombre de feature/ de ligne

    # Affichage optionnel   
    if st.checkbox('Voir des données sources (40 valeurs)'):     # chekbox optionel
        st.dataframe(df.sample(40))
    if st.checkbox('Voir le résumé'):     # chekbox optionel
        st.dataframe(df.describe())

    st.write("###### Queqlues chiffres")
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

 
    if st.checkbox('voir quelques variables'):
        options = st.multiselect(
        'Colonnes à afficher',
        df.columns)
        

        # st.write(options)
        st.write(df[options].sample(n = 20))

# Page "DataViz"
# Il faudra importer un df propre avec plus de variables ?

# Idée de graph interactif = corrélation CO2/Autre variable (distplot.. autres)