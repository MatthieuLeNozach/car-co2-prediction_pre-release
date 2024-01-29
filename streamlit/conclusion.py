import streamlit as st
import pandas as pd


# Page "Conclusion"
def app(df) :
    st.write("# Conclusion")

    st.write("""
            - Beaucoup de pistes et directions possibles (analyses, recherches complèmentaires)
             - Preprocessing important (volume, temps de travail, de recherche et de test...)
             - Des modèles efficaces pour un pb "facile"
             

             Un travail de synthèse ayant permit de pratiquer et progresser sur de nombreux sujets: 
            - Travail d'equipe, coordination, outils (Github...)
            - Beaucoup (pour nous) de Python - environs...lignes de codes, avec une initiation à la programmation "propre": Classes, objets, gestion des erreurs...
            - Les modèles et leurs spécificités
            - W d'analyse, synthèse, rédaction de rapport. 
            - Beaucoup de reflexions, malgrès un sujet qui paraissait "simple"
             
            Et toujours de nouvelles pistes mal explorées (travail de recherche non formalisé) et de curiosité sans réponse.
             - Y a t'il un modèle simpliste pouvant aider les usager (rl sur qques features). résultats de la PCA
             - Combien de données pour un modèle efficace (10^?)
             - Pourqouoi aggreger les données marche "mal"
             - Un outil pour prédire sur une partie des features ("Reflexion sur les usages")
             - Pourquoi j'ai passé deux jour de bugg avant de trouver cette virgule mal placée ;)

            MERCI POUR VOTRE ACOMPAGNEMENT ET SOUTIENT
            """)