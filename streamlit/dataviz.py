import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from PIL import Image
import plotly.express as px
import numpy as np
# ------------------------------------ Fonctions -------------------------







# ------------------------------------Page "Data exploration  & Dataviz"
def app(df) : 


    # Chargement data
    df_no_dum = st.session_state.df_no_dum


    # Affichage
    st.write("""
             # Data exploration et visualisation  
             --- 
             ## Données brutes


             20 variables quantitatives + 18 variables qualitatives.   
             8 518 000 lignes.

             """)

  


    with st.expander("Variables et valeurs manquantes avant preprocessing"):
        st.write("""
            **Liste des variables et proportion de valeurs manquantes**
        """)
        st.image("./data/nan.png")



    with st.expander("Description des 38 variables"):
        HtmlFile = open("./data/Variable1.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code, height  = 500, scrolling = True  )

    st.write("""
             ## Preprocessing
             Le dataset contient des informations sur les caractéristiques des véhicules, mais aussi de nombreuses informations inutiles sans lien avec les emissions de CO2.  
             Certaines variables etaient trop corrélées avec d'autres, et ont du être supprimées.  
             Ce travail de selection des variables s'est appuyé sur des études détaillées des corrélations lorsque c'etait necessaire.   

             Apres nettoyage et gestion des valeurs manquantes, le dataset contient:  
             **9 variables explicatives** : 5 Quantitatives, 2 Binaires, et 2 Qualitatives.  
             7 millions de lignes

             Pour pouvoir appliquer les modèles de régression linéaire:
             - Encodage one-hot (get_dummies) des 2 variables catégorielles:	
               -  Pool  -> 14 modalités.  
               -  FuelType  ->  7 modalités
             -	Normalisation standard (moyenne = 0 ; écart type = 1) de toutes les variables (pas le même ordre de grandeur).

             


             """)



    ### ------------------------------ Visualisation du dataset----------------
    # TODO: Réordonner colonnes

    st.write("## Visualisation du dataset")
    st.write("15 lignes aléatoires (reset pour changer)")

    # Créer 15 lignes aleatoires

    if 'df_sample' not in st.session_state:                
        st.session_state['df_sample'] = df_no_dum.iloc[19:39]
        # st.session_state['df_sample'] = df_no_dum.sample(15)
    df_sample = st.session_state['df_sample'] 


    # Boutons 

    col1, col2 = st.columns([0.8,0.2], gap = 'large')

    with col1:
        # Checkbox: Choisir les colonnes
        var_select = st.checkbox('choisir les variables')

    with col2:
        # Bouton reset
        if st.button("Reset"):
            st.session_state['df_sample'] = df_no_dum.sample(15)
            df_sample = st.session_state['df_sample'] 


    # Selection des features
            
    if var_select :
        colonnes = st.multiselect(
            'Colonnes à afficher',df_no_dum.columns, default = list(df_no_dum.columns))
    else: 
        colonnes = df_no_dum.columns


    # Affiche le df    
    st.dataframe(df_sample[colonnes])    




    if st.checkbox('Voir un résumé'):     # chekbox optionel
        st.dataframe(df.describe())

    st.write("""
             ## Matrice de correlation des variables quantitatives    
             """)

    dft = df[[ 'Co2', 'EnginePower','EngineCapacity', 'MassRunningOrder', 'InnovativeEmissionsReductionWltp', 'ElectricRange']]
    fig, ax = plt.subplots()
    sns.heatmap(dft.corr(), ax=ax,  annot = True, cmap='RdBu_r')
    # st.write(fig)
    st.pyplot(fig,use_container_width=True)
    st.write("""
             ## Visualisation des données   
             """)

    # Distributions des variables Quantitatives

    #sm.dislot(echantillon, fit=True, line='45')

    # Checkbox: 
    if st.checkbox('**Distributions des variables Quantitatives**'):

        fig, axes = plt.subplots(2,3, figsize=(12,8))
        axes = axes.flatten()
        for i, col in enumerate([ 'Co2', 'MassRunningOrder', 'EngineCapacity', 'EnginePower', 'InnovativeEmissionsReductionWltp', 'ElectricRange']):
            ax = axes[i]
            sns.histplot(x=df[col], bins=40, color='b',alpha=0.5, ax=ax)
            fig.subplots_adjust(hspace=0.5, wspace=0.3) 
            ax.set_title(col)
        st.write(fig) 

    # Checkbox: 
    if st.checkbox('**Boxplot Co2 - Carburant**'):
        fig,ax = plt.subplots(figsize=(3,5))
        fig = sns.catplot(x='Co2', y='FuelType', data=df_no_dum, kind='box')
        st.pyplot(fig, use_container_width=False)

        img = "./data/images/boxplot_Fuel_co2.png"
        st.image(img,
                 width = 2000,
                 use_column_width= True,
                 output_format = "PNG")
        
    st.write("""
             ## Visualisation des données   
             """)


    tab1, tab2, tab3, tab4 = st.tabs(["📈 Heatmap", "🗃 Distribution", "Proportions", "Co2 Correlations"])
    data = np.random.randn(10, 1)
    with tab1:
            
        st.write("""
            ### Matrice de correlation des variables quantitatives    
            """)

        dft = df[[ 'Co2', 'EnginePower','EngineCapacity', 'MassRunningOrder', 'InnovativeEmissionsReductionWltp', 'ElectricRange']]
        fig, ax = plt.subplots()
        sns.heatmap(dft.corr(), ax=ax,  annot = True, cmap='RdBu_r')
        # st.write(fig)
        st.pyplot(fig,use_container_width=True)

    with tab2:

        st.write("""
            ### Distribution des variables quantitatives   
            """)

                  
        # Distributions des variables Quantitatives   
        fig, axes = plt.subplots(2,3, figsize=(12,8))
        axes = axes.flatten()
        for i, col in enumerate([ 'Co2', 'MassRunningOrder', 'EngineCapacity', 'EnginePower', 'InnovativeEmissionsReductionWltp', 'ElectricRange']):
            ax = axes[i]
            sns.histplot(x=df[col], bins=40, color='b',alpha=0.5, ax=ax)
            fig.subplots_adjust(hspace=0.5, wspace=0.3) 
            ax.set_title(col)
        st.write(fig) 
          
    with tab3:

        st.write("""
            ### Proportions  
            """)
        img = "./data/images/Fuel_pie.png"
        st.image(img,
                width = 2000,
                use_column_width= True,
                output_format = "PNG")
                  
    with tab4:

        st.write("""
            ### Correlations avec CO2
            """)
        img = "./data/images/corel_co2.png"
        st.image(img,
                width = 2000,
                use_column_width= True,
                output_format = "PNG")

    st.write("""
            ### Distribution des voitures polluantes
            """)
    
    img = "./data/images/co2_distr.png"
    st.image(img,
            width = 2000,
            use_column_width= True,
            output_format = "PNG")

    
    st.write("""
            ### Distribution des voitures polluantes par type d'energie
            """)
    
    img = "./data/images/co2_distr_fueltype.png"
    st.image(img,
            width = 2000,
            use_column_width= True,
            output_format = "PNG")



        


































    x_name = 'EngineCapacity'
    y_name = 'EnginePower'
    dft = df[[x_name, y_name]]
    dft = dft.dropna(how = 'any')
    # fig, ax = plt.subplots(figsize=(8,8))
    # sns.lmplot(x=x_name, y=y_name, data=dft)
    # st.pyplot(fig) 

    def correlations(x_name, y_name, df=df):
        df = df[[x_name, y_name]]
        df = df.dropna(how = 'any')
        # Corrélation 
        correlation = df[x_name].corr(df[y_name])

        fig = plt.figure(figsize = (2,2))
        sns.lmplot(x = x_name, y = y_name, data = df, line_kws = {'color': 'red'})
        # plt.show())
        return fig
    

    x_name = 'EngineCapacity'
    y_name = 'EnginePower'
    dft = df[[x_name, y_name]]
    dft = dft.dropna(how = 'any')
    fig = correlations(x_name, y_name, df=dft)

    st.pyplot(fig)

    # if st.checkbox('**Voir la Correlation avec CO2**'):
    #     fig, axes = plt.subplots(2,3, figsize=(12,8))
    #     axes = axes.flatten()
    #     for i, col in enumerate(['EngineCapacity', 'EnginePower', 'MassRunningOrder', 'InnovativeEmissionsReductionWltp', 'ElectricRange']):
    #         ax = axes[i]
    #         dft = df[[col, 'Co2']]
    #         dft = dft.dropna(how = 'any')
    #         # st.write(dft)
    #         # st.scatter_chart(data = dft, x=col, y='Co2' )
    #         sns.relplot(x=col, y='Co2', data=dft, color='b',alpha=0.5, ax=ax)
    #         fig.subplots_adjust(hspace=0.5, wspace=0.3) 
    #         ax.set_title(col)
    #     st.pyplot(fig) 

    # st.scatter_chart(data = dft, x=x_name, y=y_name )



    # st.write(fig)






    # fig = sns.scatterplot(data = dft, x=x_name, y=y_name, )

    # st.write(fig) 




    # def correl_graph(x_name, y_name, df=df):
    #     dft = df[[x_name, y_name]]
    #     dft = dft.dropna(how = 'any')
    #     # # Corrélation 
    #     # correl = dft[x_name].corr(dft[y_name])
    #     sns.set_style("whitegrid")
    #     fig = plt.subplots(figsize = (2,2))
    #     sns.lmplot(x = x_name, y = y_name, data = df, line_kws = {'color': 'red'})
    #     # plt.show())
    #     return fig
    
    



    # def correl_graph(x_name, y_name, df=df):
    #     dft = df[[x_name, y_name]]
    #     dft = dft.dropna(how = 'any')
    #     # # Corrélation 
    #     # # correl = dft[x_name].corr(dft[y_name])
    #     sns.set_style("whitegrid")
    #     fig = plt.figure(figsize=(5,5))
    #     sns.lmplot(x = x_name, y = y_name, data = dft)        
    #     return fig
    
    # x_name = 'EngineCapacity'
    # y_name = 'EnginePower'


    # fig = correl_graph(x_name, y_name, df=df)
    # # st.write(dft2)


    # st.write(fig)






    # pears = pearsonr(x = df[x_name], y = df[y_name])
    # print("p-value: ", pears[1])
    # print("coefficient: ", pears[0])




#     if st.checkbox('voir quelques variables/CO2'):
#         options = st.multiselect(
#         'Colonnes à afficher',
#         df.columns)
        

#         # st.write(options)
#         st.write(df[options].sample(n = 20))

# # Page "DataViz"
# # Il faudra importer un df propre avec plus de variables ?

# # Idée de graph interactif = corrélation CO2/Autre variable (distplot.. autres)




#     df1 = df[[ 'Co2', 'MassRunningOrder','EngineCapacity', 'EnginePower']]
#     fig, ax = plt.subplots()
#     sns.heatmap(df1.corr(), ax=ax)
#     st.write(fig)

#     # fig = plt.figure()
#     # sns.countplot(x = 'trucd', hue = 'truc', data = df)
#     # st.pyplot(fig)
 
#     # fig = sns.displot(x = 'truc', data = df)
#     # plt.title("Distribution de qquchose")
#     # st.pyplot(fig)

#     # fig = sns.catplot(x='col1', y='col2', data=df, kind='point')
#     # st.pyplot(fig)

#     # fig = sns.lmplot(x='col5', y='col4', hue="col3", data=df)
#     # st.pyplot(fig)

#     # fig, ax = plt.subplots()
#     # sns.heatmap(df1.corr(), ax=ax)
#     # st.write(fig)
    
       # TODO: reordonner le df 