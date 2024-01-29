import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

"""
### Exemples de titre
"""

file_path = './data/'     
file_name_test = "nan_df.pkl"      # pour affichage avant dummies         
file_name_preprocessed = "c02_reduit_2e5.pkl"   

# Page "Exemples Streamlit"
def app(df) : 

    data = pd.read_pickle(file_path + file_name_test)
    st.write(data)

    data2 = pd.read_pickle(file_path + file_name_test)
    st.write(data2)

    
    st.markdown("<h3 style='position: relative; top: 50%; left: 35%; transform: translate(-30%, 0%); color: red;'>Page temporaire de test et exemples</h1>", unsafe_allow_html=True)   
    st.header("voici le header")

    ### Exemples de texte 
    st.write("### Introduction markdown titre 3")
    st.write('CALMETTES Ludovic, les autres noms')


    ### Utiliser html
    st.markdown("""
                <center>
                texte centré 
                </center>
                """, unsafe_allow_html = True)

    st.markdown("<br/>", unsafe_allow_html = True )



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
        
    container = st.container(border=True)
    with container:
        HtmlFile = open("./data/Variable1.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code, height  = 500, scrolling = True  )

    


    with st.expander("test"):
        st.write("""
            **Liste des variables et proportion de valeurs manquantes**
        """)
        var_num = df_no_dum.select_dtypes(exclude = 'object') # On récupère les variables numériques
        var_cat = df_no_dum.select_dtypes(include = 'object') # On récupère les variables catégorielles




    
       # TODO: reordonner le df 


