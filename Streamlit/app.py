# Il faut créer des fonctions à ranger daans des modules 
# Il faut créer le module "modele_reg.py" - (Avec scores et affiche)
# Il faut créer le module "import" (gestion des chargements et sauvegardes)
# Il faut créer le module "affiche"

# Le streamlit devra eter suport de présentation ? Voir idées des autres

# Todo: voir interet de st.session_state pour enregistrer le df, 
# a priori un peu plus rapide

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import intro
import dataviz
import model
import demo
import conclusion

# Fonction pour charger les données
@st.cache_data
def load_data():
    df = pd.read_pickle('./data/c02_fr&al_21_v04.pkl')
    return df

# Chargement des données
# df = load_data()                                # 2 solutions - choisir 
if 'df' not in st.session_state:
    st.session_state['df'] = load_data()
df = st.session_state.df

# Session State provides a dictionary-like interface where you can save information that is preserved between script reruns or multipage application
# st.session_state["my_key"] or st.session_state.my_key.


### Titres qui reste dans toutes les pages
"""
#### Exemple de titre qui reste dans toutes les pages, en markdown
> Voir d'autres exemples page introduction
"""


### SOMMAIRE sur le côté
st.sidebar.title("Emissions de CO2 des vehicules")
PAGES = { 
    "Présentation": intro,
    "Visualisations": dataviz,
    "Modélisation": model,
    "Démonstration": demo,
    "Conclusion et perspectives": conclusion
    }
st.sidebar.image(
        "./data/images/pollution3.png",
        width=250,
    )
selection = st.sidebar.radio("Menu", list(PAGES.keys()))
page = PAGES[selection]
page.app(df)

    # fig = sns.lmplot(x='col5', y='col4', hue="col3", data=df)
    # st.pyplot(fig)

    # fig, ax = plt.subplots()
    # sns.heatmap(df1.corr(), ax=ax)
    # st.write(fig)

# Page modelisation
    # fit longs (a mesurer). Donc il faudrait des modeles déja entrainés. (gestion de "data_modeles")
    # On peut faire de l'interactif sur la durée de l'entrainement, ...? (afficher la durée...)

if page == 3 : 
    st.write("### Modélisation")

# Import d'un modèle entrainé
# Import d'un X_train, y_train
    # Entrer un nouveau vehicule en choisissant les parametres --> Prédiction
    # Choix d'un véhicule dans la base de donnée, (propositions a partir de criteres fixés, ou véhicule "le plus proche") 
    # --> Prédiction / vrai valeur
    # Peut on implementer un predictif partiel, a partir de quelques (ou une) caractéristiques. --> Donne le Co2 ou son intervalle ou la proba de classe (plutot stat alors)
# Montrer les quelques graphiques réalisés (quelle interactivité ? plutot base de donnée d'images)




y = df['Co2']
X = df.drop(['Co2', 'Co2Grade', 'FuelConsumption'], axis=1)

# # Inserer dans fichier d'aide: Remplir les valeurs manquante, méthode rapide
#     for col in X_cat.columns:   
#         X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
#     for col in X_num.columns:
#         X_num[col] = X_num[col].fillna(X_num[col].median())
#     X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
#     X = pd.concat([X_cat_scaled, X_num], axis = 1)

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
#     X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# # Inserer dans fichier d'aide: tester plusieurs clf, méthode rapide
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.svm import SVC
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import confusion_matrix

#     def prediction(classifier):
#         if classifier == 'Random Forest':
#             clf = RandomForestClassifier()
#         elif classifier == 'SVC':
#             clf = SVC()
#         elif classifier == 'Logistic Regression':
#             clf = LogisticRegression()
#         clf.fit(X_train, y_train)
#         return clf

#     def scores(clf, choice):
#         if choice == 'Accuracy':
#             return clf.score(X_test, y_test)
#         elif choice == 'Confusion matrix':
#             return confusion_matrix(y_test, clf.predict(X_test))
        
#     choix = ['Random Forest', 'SVC', 'Logistic Regression']
#     option = st.selectbox('Choix du modèle', choix)
#     st.write('Le modèle choisi est :', option)

#     clf = prediction(option)
#     display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
#     if display == 'Accuracy':
#         st.write(scores(clf, display))
#     elif display == 'Confusion matrix':
#         st.dataframe(scores(clf, display))


