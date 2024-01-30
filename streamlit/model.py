import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definition des chemins et noms 
images_path = './data/images/'     


# Page "Modeles de ML"
def app(df) :
    st.write("# Algorithmes de Machine Learning")

    st.write("""
            Nous avons expérimentés plusieurs modèles, le principe étant:  
            1. De tester différents algorithmes et méthodes en augmentant progressivement leur complexité, pour comparer leurs performances (scores, robustesse, vitesse). &nbsp;&nbsp; => &nbsp;&nbsp;    Selection du ou des meilleurs modèles  
            2. De chercher à améliorer le (ou les) modèle final retenu (optimisation des paramètres)  
            3. Enfin, de s'interesser à l'interprétabilité et la  "feature importance" de ces modèles finaux, pour analyse.
             
            Nous avons mené cette démarche à la fois pour la **régression** et la **classification**. 
            """)



    tab1, tab2, tab3, tab4 = st.tabs(["regression", "classification", "deep learning", "interprétabilité"])

    ####################################### REGRESSIOIN ##############################
    with tab1:

        st.write("""
                # Un problème de Régression
                ---
                ## 1. Modèle de Regression Linéaire 
                     """)
        
        col1, col2 = st.columns([0.62, 0.38], gap = 'small')
        with col1:
            st.write("""
                - ###### Modèle simple sans régularisation
                - ###### Modèle régularisé avec recherche des meilleurs paramètres.
                Même si notre premier modèle ne semblait pas conduire à du surapprentissage, nous avons voulu tester differents paramètres de régularisation \
                afin de le confirmer, et observer les effets des régularisations (Lasso, Ridge).  
                
                **La régularisation n'apporte rien !!**  
                Les meilleurs paramères trouvés par grille de recherche et valdation croisée sont pour alpha = 0, c'est a dire sans régularisation.   
                """)
            img_name="form_regr.png"
            st.image(images_path + img_name, use_column_width= "auto" )
        with col2:
            img_name = "lr_elastic_net.png"    
            st.image(images_path + img_name,
            use_column_width= True )
        
        st.write("""
                    #### Résultats :  """)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.write("""   
                    Les scores obtenus sur l'échantillon d'apprentissage et l'échantillon de test sont très proches et ne montrent pas de sur-apprentissage. Ce que confirme une validation croisée.  
                    L'erreur moyenne sur les prédictions est d'environ 10g/km pour des valeurs de l'ordre de 130 g/km.  
                    L'erreur moyenne relative est d'environ 10%  
                    """)
        with col2:
            img_name = "score_lr.png"     
            st.image(images_path + img_name,
            use_column_width= True )

        st.write("Ce modèle donne déjà des résultats qui semblent corrects avec un **R2_score de 0,89**, c’est-à-dire que 89% de la variance du CO2 peut être expliquée par ce modèle de régression.")
        st.write("")  

        st.write("## 2. Modèle XGBoost pour la Régression.")  

        col1, col2 = st.columns([0.65, 0.35], gap = 'medium')
        with col1:             
            st.write("""
                    XGBoost est une amélioration optimisée de l'algorithme de boosting en arbres de décision.
                    Le boosting consiste à entraîner plusieurs modèles faibles (ici des arbres de décision peu profonds) de manière itérative, en mettant à chaque itération l'accent sur les erreurs \
                    commises par les arbres précédents. Le modèle final tient compte de l'ensemble des modèles faibles entrainés pour fournir ses prédictions. 
                    """)
            
            xgb_param = st.expander("Recherche des meilleurs parametres" , expanded=False)
            with xgb_param:
                st.write("""
                    ###### L'entraînement du modèle XGBoost a été effectué avec des paramètres variables que l'on a cherché à optimiser :                   
                    - **learning_rate** :   
                        'Taux d’apprentissage'. Une valeur faible entraîne un modèle plus robuste au sur-apprentissage, mais un calcul et une convergence plus lents qui nécessitent plus d'itérations. 
                        Testé de 1 à 0,1.  
                    - **max_depth** :  
                        Profondeur des arbres. Plus les arbres sont profonds, plus le modèle est com-plèxe et plus grandes sont les chances d'overfitting. Testé 6 ± 2  
                    - **colsample_bytree** :  
                        Fraction de caractéristiques (features) à utiliser lors de la construction de chaque arbre de décision. Permet de gérer l'overfitting. Testé de 1 à 0,4  
                    - **num_boost_round** :  
                        Nombre maximum d'itérations ou de boosters (arbres) à entraîner. Ajusté en fonction des autres paramètres pour avoir une convergence.  
                    - **early_stopping_rounds** :  
                        Arrêt anticipé de l'entraînement si aucune amélioration significative n'est observée pendant un certain nombre d'itérations défini par ce paramètre. Entre 15 et 20.
                    
                    Des performancessouvent été très proches et très bonnes. 
                    Le premier modèle testé avec les paramètres par défaut avait un score RMSE de 3,06 et les meilleurs modèle avec un Learning rate entre 0,1 et 0,5 ont un score RMSE de 2,42. 
                        """)
            

        with col2:
            url_image = "https://www.researchgate.net/profile/Li-Mingtao-2/publication/335483097/figure/fig3/AS:934217085100032@1599746118459/A-general-architecture-of-XGBoost.ppm"
            st.image(url_image, width = 300, use_column_width= True )
        

        st.write("""
                #### Résultats :  """)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.write("""   
                **On obtient d'excellents scores**:   
                Pas de sur-apprentissage (testé). L'erreur moyenne sur les prédictions est d'environ 1g/km pour des valeurs de l'ordre de 130 g/km.  
                L'erreur moyenne relative est d'environ 1 %  

                Ce modèle donne donc d'excellents résultats avec un **R2_score de 0,997**.   
                99,7 % de la variance du CO2 peut être expliquée par le modèle. Difficile d'imaginer faire mieux !

                    """)
        with col2:
            img_name = "score_xgb.png"      
            st.image(images_path + img_name,
            use_column_width= True )

        st.write("## Analyse plus détaillées des résultats XGBoost") 
        st.write("")




        col1, col2, col3 = st.columns([0.26, 0.35, 0.39], gap = 'small')
        img_name="reg_xgb_ex.png" 
        col1.image(images_path + img_name, use_column_width= True )
        img_name="reg_xgb_plot2.png"
        col2.image(images_path + img_name, use_column_width= True )
        img_name="reg_xgb_residus.png" 
        col3.image(images_path + img_name, use_column_width= True )
        compare = col3.checkbox("comparer avec la regression lineaire")

   
        if compare :
            col1, col2, col3 = st.columns([0.26, 0.35, 0.39], gap = 'small')
            img_name="reg_lr_ex.png" 
            col1.image(images_path + img_name, use_column_width= True )
            img_name="reg_lr_plot.png"
            col2.image(images_path + img_name, use_column_width= True )
            img_name="reg_lr_residus.png" 
            col3.image(images_path + img_name, use_column_width= True )        

        st.write("Distribution des résidus")
        col1, col2 = st.columns([0.35, 0.65], gap = 'medium')
        img_name="reg_xgb_qq.png"  
        col1.image(images_path + img_name, use_column_width= True )
        img_name=img_name="reg_xgb_box.png"
        col2.image(images_path + img_name, use_column_width= True )

        st.write("")
         
        st.write("""   
                ## Bilan modèles de regression :
                - Excellent score pour XGBoost, erreur moyenne de 1g/km (1 %)
                 - Regression linéaire : peut aider à comprendre simplement les features, le modèle XGBoost est plus difficule à interpréter (Modele d'ensemble)
                 """)
        
    ####################################### CLASSIFICATION ##############################
        
        
        with tab2:
            st.write("# Un problème de classification")  
            col1, col2 = st.columns([0.75, 0.25])
            with col1:
                st.write("""
                    La classification consiste à prédire l'apparteance d'un véhicule à l'un des 7 scores CO2.
                    Les valeurs continues des émissions de CO2 sont donc discrétisées en 7 classes. 
                    """)
            with col2:
                img_name = "co2score.png"
                st.image(images_path + img_name, width=200)
                
            st.write("## Modèles de classification par arbre(s) de décision")  
         
            col1, col2 = st.columns([0.65, 0.35])
            with col1:
                st.write("### 1. Le Decision Decision Tree Classifier")  
                st.write("""
                         C'est l'un des classificateurs les plus simples et faciles à expliquer à un public non initié.
                         Il consiste à diviser le jeu de données en sous-ensembles de plus en plus homogènes selon une valeur de séparation.
                         Ils ne nécessitent pas de transformation de type normalisation ou standardisation des données.
                         """)
                dtc_param = st.expander("Recherche des meilleurs parametres" , expanded=False)
                with dtc_param:
                    st.write("""
                        ###### Hyperparamètres du Decision Tree Classifier:                   
                        - **criterion** : Critère de qualité de la séparation. On utilise l'indice de Gini.   
                        - **max_depth** :  Le nombre maximal de séparations à effectuer. 
                        Un grand nombre de séparations peut conduire à du sur-apprentissage. 
                        Nous avons observé des performance optimales pour une profondeur de 20.
                            """)
            with col2:
                img_name = "tree.png"
                st.image(images_path + img_name, use_column_width= True )

                    
            
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.write("### 2. Le Random Forest Classifier")  
                st.write("""
                         C'est un modèle basé sur un ensemble d'arbres de décision de dimension réduite.
                         Chacun des estimateurs opère sur un nombre réduit de features et vote pour l'appartenance à une classe.
                         La classe ayant eu la majorité des votes sur une ligne à prédire devient la prédiction.
                         Les Random Forest sont des modèles performants et réputés bien plus robustes qu'un arbre de décision classique.
                         """)
                dtc_param = st.expander("Recherche des meilleurs parametres" , expanded=False)
                with dtc_param:
                    st.write("""
                        ###### Hyperparamètres du Random Forest Classifier:                   
                        - **n_estimators** : Le nombre d'arbres (estimateurs) créés. Un nombre trop élevé d'estimateurs peut aussi conduire à du sur-apprentissage.   
                        Un grand nombre de séparations peut conduire à du sur-apprentissage. 
                        Nous avons opté pour 200 estimateurs. 
                        - **max_features**: Le nombre de features à utiliser pour chaque estimateur. la valeur 'sqrt' sera utilisée, il s'agit de limiter le nombre de features à la racine carrée du nombre total de features. 

                            """)
            with col2:
                img_name = "Screenshot from 2024-01-29 19-39-02.png"
                st.image(images_path + img_name, use_column_width= True )
                
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.write("### 2. Le XGBoost Classifier")  
                st.write("""
                         C'est l'adaptation du modèle présenté précedemment pour les problèmes de classification.
                         Il a une approche similaire au Random Forest, mais les estimateurs sont ajustés aux erreurs de l'estimateur précédent plutôt que soumis à un vote.
                         """)
                dtc_param = st.expander("Recherche des meilleurs parametres" , expanded=False)
                with dtc_param:
                    st.write("""
                        ###### Hyperparamètres du XGBoost Classifier:                   
                        - **objective** : On utilisera *multi:softmax* pour répondre au prblème de classification multiclasse. La fonction softmax est utilisée pour calculer les probabilités d'appartenance à chaque classe, et la classe ayant la probabilité la plus élevée est choisie.
                        - **n_estimators**: Dans le cas du xgboost, il s'agit du nombre d'epochs, ou nombre d'estimateurs faibles à qui l'on donne toutes les données d'entraînement. 
                            """)
            with col2:
                img_name = "Screenshot from 2024-01-29 22-15-39.png"
                st.image(images_path + img_name, width=300)
            
            st.write("## Analyse des résultats")  
            st.write(""" """)
            col1, col2, col3 = st.columns([0.34, 0.33, 0.33])


            col1, col2, col3 = st.columns([0.34, 0.33, 0.33])

            with col1:
                
                #st.markdown("<br>"*2, unsafe_allow_html=True)  # Add empty space before the image
                img_name = "Screenshot from 2024-01-29 20-21-10.png"
                st.image(images_path + img_name, use_column_width=True)
                img_name = "Screenshot from 2024-01-29 20-18-07.png"
                st.image(images_path + img_name, use_column_width=True)

                
            with col2:
                img_name = "Screenshot from 2024-01-29 20-25-42.png"
                st.image(images_path + img_name, use_column_width=True)
                img_name = "Screenshot from 2024-01-29 20-06-20.png"     
                st.image(images_path + img_name, use_column_width= True )
                
            with col3:
                img_name = "Screenshot from 2024-01-29 21-06-46.png"
                st.image(images_path + img_name, use_column_width=True)
                img_name = "Screenshot from 2024-01-29 21-09-19.png"     
                st.image(images_path + img_name, use_column_width= True )

            st.write("")  
            


            st.write("""   
                **On obtient d'excellents scores**:   
                Dans les 3 cas, les modèles par arbre logique donnent des résultats similaires.  
                Quasiment toutes les erreurs de classes se trouvent sur une classe adjacente. 
                On observe légèrement plus d'erreurs sur la classe D.
                La classe a est quasiment parfaitement prédite, possiblement dû la facilité qu'ont modèles à augmenter la pureté après avoir séparé véhicules hybrides et électriques. 

                    """)


            st.write("## Analyse des feature importances") 
            st.write("")




            col1, col2, col3 = st.columns([0.26, 0.35, 0.39], gap = 'small')
            with col1:
                st.write("**Decision Tree: premier split**")
                st.markdown("<br>"*3, unsafe_allow_html=True)  # Add empty space before the image
                img_name="Screenshot from 2024-01-29 20-00-17.png" 
                col1.image(images_path + img_name, use_column_width= True )
            with col2:
                st.write("**Random Forest: feature importance**")
                st.markdown("<br>"*2, unsafe_allow_html=True)  # Add empty space before the image
                img_name="Screenshot from 2024-01-30 10-56-36.png"
                col2.image(images_path + img_name, use_column_width= True )
            with col3:
                st.write("**XGBoost: feature importance**")
                img_name="Screenshot from 2024-01-30 11-10-02.png" 
                col3.image(images_path + img_name, use_column_width= True ) 




            st.write("")
            
            st.write("""   
                    ## Bilan modèles de machine learning pour la classification:
                    - Excellents scores pour les 3 modèles, mais XGBoost et RFC promettent plus de robustesse
                    """)
                         


       
    ####################################### DEEP LEARNING ###############################
        
        with tab3:
            
            st.write("## Classification par Deep Learning (Réseau de neurones)")  
         
            col1, col2 = st.columns([0.65, 0.35])
            with col1:
                st.write("### Présentation du DNN ou Deep Neural Network")  
                st.write("""
                            Présentation rapide DNN de classification BLABLABLA
                         """)
                dtc_param = st.expander("Recherche des meilleurs parametres" , expanded=False)
                with dtc_param:
                    st.write("""
                        ###### Paramétrage d'un réseau de neurones:                   
                        - **Couche d'entrée:** BLABLA
                        - **Couches cachées:** BLABLABLA   
                        - **Neurones par couche** :  BLABLA
                        - **Régularisation ADAM** :  BLABLA 
                        - **Fonctions d'activation** :  ReLU BLABLA , Softmax BLABLA
                            """)
            with col2:
                img_name = "Dense-Neural-Network.png"
                st.image(images_path + img_name, use_column_width= True )

            
        
            col1, col2, col3 = st.columns([0.333, 0.333, 0.333], gap = 'small')
            with col1:
                
                #st.markdown("<br>"*2, unsafe_allow_html=True)  # Add empty space before the image
                st.write("**Architecture simple**")
                st.write("Validation Accuracy = 0.9231")
                img_name = "Screenshot from 2024-01-30 11-22-55.png"
                st.image(images_path + img_name, use_column_width=True)

                
            with col2:
                st.write("**Architecture profonde**")
                st.write("Validation Accuracy: 0.881")
                img_name = "Screenshot from 2024-01-30 11-24-52.png"
                st.image(images_path + img_name, use_column_width=True)
                
            with col3:
                st.write("**Architecture large**")
                st.write("Validation Accuracy = 0.9235")
                img_name = "Screenshot from 2024-01-30 11-27-17.png"
                st.image(images_path + img_name, use_column_width=True)

            
            st.write("### Interpétation des résultats")  

            st.write("""Modèles pas plus performlants que le ML mais coût computationnel plus élevé
                     BLABLABLABLA
                     """)
            st.write("")  


    ####################################### INTERPRETABILITE #############################
                 
    with tab4:

        st.write("""
                # Interprétabilité des modèles
                ---
                 Un modèle très performant: XGBoost => Peut on comprendre ses décicions ?
                 - feature importance
                 - shap_values
                 - arbre de décission
                 """)
        
        st.write('## Calcul de la "Feature Importance"')
        st.write("")
        # Texte a centrer
        st.write("##### Méthode XGBoost")
        img_name="xgb_feat1.png"
        st.image(images_path + img_name, use_column_width= "auto" )

        st.write("##### Méthode Shap Values")
        col1, col2 = st.columns([0.65, 0.35], gap = 'medium')
        img_name="xgb_feat2.png"
        col1.image(images_path + img_name, use_column_width= True )

        st.write("##### Méthode Skater")
        img_name="xgb_feat3.png"
        st.image(images_path + img_name, use_column_width= True )

        st.write("""
                - Des résultats assez différents.  
                - La feature importance n'est pas un concept mathématique définit, il existe plusieurs approches pour l'évaluer.  
                - Sur les 6 premières valeurs, 5 de commune. => **Les 5 plus influentes**  
                  Plus de 82% de la feature importance totale, quelquesoit la méthode.
                    -	Electric Range
                    -	Mass
                    -	Engine Power
                    -	Engine Capacity
                    -	Innovative Emission WLTP
                 
                 ## Les "Shap_Values" - Méthode retenue             
                SHAP (SHapley Additive exPlanations) est une technique d'interprétabilité des modèles basées sur la théorie des jeux et la théorie des ensembles.  
                Les shap values possèdent des propriétés mathématiques cohérentes => Renforce leur crédibilité et usages.  
                Des algorithmes performants et de nombreux outils graphiques   
                Sur notre modèle XGBoost de regression, cela parait être la méthode la plus "logique" (fariquants "mal classés", petrol et diesel "importants").   
                
                Les émissions de CO2 sont donc principalement liées à: 
                -	La présence d'un mode électriques et son autonomie. 				37 %
                -	Le poids du véhicule.								23 %
                -	Les caractéristiques moteur (cylindrée et puissance).				21 %
                -	La présence et l'efficacité de technologies innovantes de réduction de CO2.	4 %	
                 
                ## Les "Shap_Values" - Outils graphiques

                **Interprétation globale**
                 

                 """)

        col1, col2 = st.columns([0.5, 0.5], gap = 'medium')
        img_name="xgb_shap1.png"
        col1.image(images_path + img_name, use_column_width= "auto" )

        img_name="xgb_shap2.png"
        col2.image(images_path + img_name, use_column_width= "auto" )
        col2.write("")

        st.write("**Interprétation locale**")
        col1, col2 = st.columns([0.8, 0.2], gap = 'medium')
        img_name="xgb_shap3bis.png"
        col1.image(images_path + img_name, use_column_width= "auto" )




   

        st.write('## Arbre de décision"')
        st.write('##### XGB: Arbre de décision de rang 0, sur 4 niveau"')
        st.write("""
                    Il ne permet que d'avoir une représentation simpliste de l'algorithme utilisé par XGBoost.  
                    Dans la réalité pour notre modèle, il se combine avec 499 autres arbres de poids moindre, et sa profondeur va jusqu’à 6.""")
        
        img_name="xgb_plot_tree.png"
        st.image(images_path + img_name, use_column_width= True )

                 
  
                 


####################################### RELIQUAT A LAISSER ##############################
# Pour le moment (peut etre à utiliser)


        #         ## 1. Modèle de Regression Linéaire 
        #              """)
        
        # col1, col2 = st.columns([0.62, 0.38], gap = 'small')
        # with col1:
        #     st.write("""
        #         - ###### Modèle simple sans régularisation
        #         - ###### Modèle régularisé avec recherche des meilleurs paramètres.
        #         Même si notre premier modèle ne semblait pas conduire à du surapprentissage, nous avons voulu tester differents paramètres de régularisation \
        #         afin de le confirmer, et observer les effets des régularisations (Lasso, Ridge).  
                
        #         **La régularisation n'apporte rien !!**  
        #         Les meilleurs paramères trouvés par grille de recherche et valdation croisée sont pour alpha = 0, c'est a dire sans régularisation.   
        #         """)
            
        # with col2:
        #     img_name = "lr_elastic_net.png"    
        #     st.image(images_path + img_name,
        #     use_column_width= True )

        # st.write("")

        # texte_colore = ":grey[comparer avec la regression lineaire]"
        # comparer = st.expander(texte_colore , expanded=False)
        # with comparer:
        #     col1, col2, col3 = st.columns([0.26, 0.35, 0.39], gap = 'small')
        #     img_name="reg_lr_ex.png" 
        #     col1.image(images_path + img_name, use_column_width= True )
        #     img_name="reg_lr_plot.png"
        #     col2.image(images_path + img_name, use_column_width= True )
        #     img_name="reg_lr_residus.png" 
        #     col3.image(images_path + img_name, use_column_width= True )



        # texte_colore = ":grey[comparer avec la regression lineaire]"
        # comparer2 = st.expander("<div style='text-align:right;'>Texte aligné à droite</div>", expanded=False)
        # with comparer2:
        #     st.markdown("<div style='text-align:right;'>Texte aligné à droite</div>", unsafe_allow_html=True)
        #     col1, col2 = st.columns([0.35, 0.65], gap = 'large')
        #     img_name=img_name="reg_lr_box.png"
        #     col2.image(images_path + img_name, use_column_width= True )





                # st.write("""
                #         XGBoost est une amélioration optimisée de l'algorithme de boosting en arbres de décision.
                #         Le boosting consiste à entraîner plusieurs modèles faibles (ici des arbres de décision peu profonds) de manière itérative, en mettant à chaque itération l'accent sur les erreurs \
                #         commises par les arbres précédents. Le modèle final tient compte de l'ensemble des modèles faibles entrainés pour fournir ses prédictions. 
                #         """)

            # # st.write("Ce modèle donne déjà des résultats qui semblent corrects avec un **R2_score de 0,89**, c’est-à-dire que 89% de la variance du CO2 peut être expliquée par ce modèle de régression.")
            # # st.write("")  
             
            




                 







                  
                  

          


    # tab2.subheader("A tab with the data")
    # tab2.write(data)




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
#         st.dataframe(scores(clf, display)