import streamlit as st
import pandas as pd
import pickle
import joblib
import xgboost as xgb


# Page "Demo"
def app(df) : 

#   -----------------------------------Fonctions et code necessaire-----------

    # Charge model

    filepath = "./data"
    filename = "xgb1901.joblib"
    adress = f'{filepath}/{filename}'
    model = joblib.load(adress)


    # Separate variables from target for regression

    def reg_prepross(df):
        X = df.drop(['Co2', 'Co2Grade', 'FuelConsumption'], axis=1)
        y = df['Co2']
        return (X,y)


    # Initialiser un df_ligne avec la moyenne, fonction si aussi utilisée pour créer la ligne de test

    def create_X_ligne(CategoryOf, MassRunningOrder, EngineCapacity, EnginePower, InnovativeTechnology,
                    InnovativeEmissionsReductionWltp, ElectricRange, Pool, FuelType):
        """ 
        exemple: create_X_ligne(1500, 1600, 110, 0, 0, 0, 'STELLANTIS', 'PETROL') 
        """
        # Extraire une ligne sous forme de df
        df_ligne = df.iloc[2:3]
        # Séparer X et y pour une ligne
        (X_ligne, y_ligne) = reg_prepross(df_ligne) 
        X_ligne['CategoryOf'] = CategoryOf               # Moy       # min/max   
        X_ligne['MassRunningOrder'] = MassRunningOrder   # 1474      # 915/3275
        X_ligne['EngineCapacity'] = EngineCapacity       # 1570      # 875/6750
        X_ligne['EnginePower'] = EnginePower             # 110       # 44/574
        X_ligne['InnovativeTechnology'] = InnovativeTechnology
        X_ligne['InnovativeEmissionsReductionWltp'] = InnovativeEmissionsReductionWltp       # 1.15   # 0/2.9
        X_ligne['ElectricRange'] = ElectricRange         # 6    # 0/126

        # Remplir les dummies de 0
        colonnes_pool = [col for col in X_ligne.columns if col.startswith('Pool')]
        colonnes_fuel = [col for col in X_ligne.columns if col.startswith('Fuel')]
        X_ligne[colonnes_pool+colonnes_fuel] = 0

        # Faire un agrégat pour récupérer le nom des dummies, et mettre à 1
        pool_column = "Pool_"+ Pool                      # STELLANTIS         
        fuel_column = "Fuel_"+ FuelType                  #'PETROL'  
        X_ligne[[pool_column, fuel_column]] = 1

        return X_ligne






# -------------------AFFICHAGE-------------------------------

    st.write("# Demonstration")

    # ----- Prediction du niveau de CCO2----------------------------------------------------

    st.write("""
            #### Outil pour prédire le niveau de CO2 émis d'un vehicule:  
            """)


    col1, col2 = st.columns([0.48,0.52], gap = 'medium')
    with col1:

        # Interface pour récuperer les valeurs choisies par l'utilisateur

        # Prevoir un bouton de choix pour rentrer l'un, l'autre ou les deux. 
        # Faire des choix pareil pour les autres, soit en utilisant une corrélation, soit par valeur moyenne (boutons de choix + quelle corrélation ou calcul automatique de correlation)   


        FuelType = st.selectbox('Fuel type',
                    ['PETROL',
                        'DIESEL',
                        'ELECTRIC',
                        'PETROL/ELECTRIC',
                        'DIESEL/ELECTRIC',
                        'LPG',
                        'ETHANOL',
                        'NATURALGAS',
                        'HYDROGEN'              
                        ],index = 0 )
    


        EngineCapacity = st.slider('EngineCapacity', 850, 7000, 1600)
        EnginePower = st.slider('EnginePower (unité)', 40, 600, 110)      

        cond = (FuelType in ['PETROL/ELECTRIC', 'DIESEL/ELECTRIC'])
        ElectricRange = st.slider('Autonomie en mode electrique (km)', 10, 150, 0, disabled = (cond == False))
        # else:
        #     ElectricRange = 0


    with col2:   

        Pool = st.selectbox('Pool',
                            ['STELLANTIS',
                            'VW-SAIC',
                            'RENAULT-NISSAN-MITSUBISHI',
                            'MAZDA-SUBARU-SUZUKI-TOYOTA',
                            'HYUNDAI',
                            'KIA',
                            'BMW',
                            'TESLA',
                            'FORD',
                            'MERCEDES-BENZ',
                            'VOLKSWAGEN',
                            'HYUNDAI MOTOR EUROPE',
                            'TESLA-HONDA-JLR'])
        
        MassRunningOrder = st.slider('Masse du véhicule (kg)', 900, 3500, 1500)
        
        category = st.checkbox('Vehicule tout terrain')
        if category == True:
            CategoryOf = 1
        else : 
            CategoryOf = 0 
        IT = st.checkbox('Innovative Technology')
        if IT == True:
            InnovativeTechnology = 1
            InnovativeEmissionsReductionWltp = st.slider("Valeur de l'Innovative Emissions Reduction",float(0), 2.9, 1.15) 
        else:
            InnovativeTechnology = 0
            InnovativeEmissionsReductionWltp = 0  




    col1, col2 = st.columns([0.25,0.75])

    with col1:
        co2 = st.button('**Predire le niveau de CO2**')

        if co2:
            X_ligne = create_X_ligne(CategoryOf, MassRunningOrder, EngineCapacity, EnginePower, InnovativeTechnology, InnovativeEmissionsReductionWltp, ElectricRange, Pool, FuelType)
            # Le preprocessing attendu pour en faire un X_test
            test = xgb.DMatrix(X_ligne)
            ypred = model.predict(test)[0]
            
            # Catégorie decilles (fr-al 21: 10/25/50/75/90 = 0/62/124/142/167) 
            if ypred < 62 :
                text = "Félicitations. Cette voiture devrait etre dans les 25% les moins polluantes. Seule les electriques sont dans les 10%" 
            elif ypred < 124 :
                text = "Pollution moyenne (2nd quartille)"
            elif ypred < 142 :
                text = "Pollution assez forte(3eme quartille)"
            elif ypred < 167:
                text = "Pollution forte(4eme quartille)"
            else :
                text = "Felicitation. Cette voiture devrait etre dans les 10% les plus polluantes. On peut pas faire mieux !!"

    with col2:
        if co2:
            st.write(f"Votre vehicule emettra : **{ypred:.2f} g/km**")
    if co2:
        st.write(text)



    # with col2:
    #     if co2:




            #  st.write("\n")
            # st.write("\n")
            # st.write("\n")
    
    # st.write("\n")
    # st.write("\n")
    