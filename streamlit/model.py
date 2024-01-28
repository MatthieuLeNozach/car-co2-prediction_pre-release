import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import d'un modèle entrainé
# Import d'un X_train, y_train
# Entrer un nouveau vehicule en choisissant les parametres --> Prédiction
# Choix d'un véhicule dans la base de donnée, (propositions a partir de criteres fixés, ou véhicule "le plus proche") 
# --> Prédiction / vrai valeur
# Peut on implementer un predictif partiel, a partir de quelques (ou une) caractéristiques. --> Donne le Co2 ou son intervalle ou la proba de classe (plutot stat alors)
# Montrer les quelques graphiques réalisés (quelle interactivité ? plutot base de donnée d'images)


# Page "Model"
def app(df) :
    st.write("### Modélisation")

    st.write("""
            Nous avons expérimentés de nombreux modèles, le principe de notre démarche etait:  
            1. De tester différents modèles en augmentant progressivement leur complexité, afin de comparer les performances, la stabilité et la vitesse. &nbsp;&nbsp; => &nbsp;&nbsp;    Selection du ou des meilleurs modèles  
            2. De chercher à améliorer le ou les modèles finaux retenu (optimisation des parametres)  
            3. Enfin, de s'interesser a la "feature importance" et interprétabilité de ces modèles finaux, pour analyse.
             
            Nous avons mené cette démarche à la fois sur des modèles de régression et de classification. 
            """)



    tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
    data = np.random.randn(10, 1)
    tab1.subheader("A tab with a chart")
    with tab1:
            
        st.write("""
            ## Matrice de correlation des variables quantitatives    
            """)

        dft = df[[ 'Co2', 'EnginePower','EngineCapacity', 'MassRunningOrder', 'InnovativeEmissionsReductionWltp', 'ElectricRange']]
        fig, ax = plt.subplots()
        sns.heatmap(dft.corr(), ax=ax,  annot = True, cmap='RdBu_r')
        # st.write(fig)
        st.pyplot(fig,use_container_width=True)

    with tab2:
                  
                  
        fig, axes = plt.subplots(2,3, figsize=(12,8))
        axes = axes.flatten()
        for i, col in enumerate([ 'Co2', 'MassRunningOrder', 'EngineCapacity', 'EnginePower', 'InnovativeEmissionsReductionWltp', 'ElectricRange']):
            ax = axes[i]
            sns.histplot(x=df[col], bins=40, color='b',alpha=0.5, ax=ax)
            fig.subplots_adjust(hspace=0.5, wspace=0.3) 
            ax.set_title(col)
        st.write(fig) 
          


    tab2.subheader("A tab with the data")
    tab2.write(data)

"""
Construire une page qui permet de choisir toutes les caractériqtiques, et de préadire le CO2 (reglin)


"""


    # def lr(Engine_Size, Cylinders, Fuel_Consumption_City,Fuel_Consumption_Hwy, Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg,Fuel_Type_E, Fuel_Type_X,Fuel_Type_Z, Transmission_A4, Transmission_A5, Transmission_A6,Transmission_A7, Transmission_A8, Transmission_A9,Transmission_AM5, Transmission_AM6, Transmission_AM7,Transmission_AM8, Transmission_AM9, Transmission_AS10,Transmission_AS4, Transmission_AS5, Transmission_AS6,Transmission_AS7, Transmission_AS8, Transmission_AS9,Transmission_AV, Transmission_AV10, Transmission_AV6,Transmission_AV7, Transmission_AV8, Transmission_M5,Transmission_M6, Transmission_M7, Make_Type_Luxury,Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV,Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck):
    # c=pd.DataFrame([Engine_Size, Cylinders, Fuel_Consumption_City,Fuel_Consumption_Hwy, Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg,Fuel_Type_E, Fuel_Type_X,Fuel_Type_Z, Transmission_A4, Transmission_A5, Transmission_A6,Transmission_A7, Transmission_A8, Transmission_A9,Transmission_AM5, Transmission_AM6, Transmission_AM7,Transmission_AM8, Transmission_AM9, Transmission_AS10,Transmission_AS4, Transmission_AS5, Transmission_AS6,Transmission_AS7, Transmission_AS8, Transmission_AS9,Transmission_AV, Transmission_AV10, Transmission_AV6,Transmission_AV7, Transmission_AV8, Transmission_M5,Transmission_M6, Transmission_M7, Make_Type_Luxury,Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV,Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck]).T
    # return model.predict(c)
          
    
    # prediction=lr(Engine_Size, Cylinders, Fuel_Consumption_City,Fuel_Consumption_Hwy, Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg,Fuel_Type_E, Fuel_Type_X,Fuel_Type_Z, Transmission_A4, Transmission_A5, Transmission_A6,Transmission_A7, Transmission_A8, Transmission_A9,Transmission_AM5, Transmission_AM6, Transmission_AM7,Transmission_AM8, Transmission_AM9, Transmission_AS10,Transmission_AS4, Transmission_AS5, Transmission_AS6,Transmission_AS7, Transmission_AS8, Transmission_AS9,Transmission_AV, Transmission_AV10, Transmission_AV6,Transmission_AV7, Transmission_AV8, Transmission_M5,Transmission_M6, Transmission_M7, Make_Type_Luxury,Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV,Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck)
    # return render_template('index.html',prediction_text="Co2 Emissions by car is {}".format(np.round(prediction,2)))

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