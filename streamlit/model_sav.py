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
            img_name="form_regr.png"
            # st.image(images_path + img_name, width = 400, use_column_width= "never" )
            st.write("""
                - ##### Modèle simple sans régularisation
                - ##### Modèle régularisé avec recherche des meilleurs paramètres.
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

            st.write("""
                # Un problème de Classification 
                ---
                 Particularités :
                 Modèles explorés :
                 - feature importance
                 - shap_values
                 - arbre de décission

                 """)


        
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
                La feature importance n'est pas un concept mathématique définit, il existe plusieurs approches pour l'évaluer.  
                - Sur les 6 premières valeurs, 5 de commune. => **Les 5 plus influentes**  
                  Plus de 82% de la feature importance totale, quel que soit la méthode.
                    -	Electric Range
                    -	Mass
                    -	Engine Power
                    -	Engine Capacity
                    -	Innovative Emission WLTP
                 
                 ## Les Shap_Values - Méthode retenue             
                SHAP (SHapley Additive exPlanations) est une technique d'interprétabilité des modèles basées sur la théorie des jeux et la théorie des ensembles.  
                Les shap values possèdent des propriétés mathématiques cohérentes => Renforce leur crédibilité et usages.  
                Des algorithmes performants et de nombreux outils graphiques.   
                Sur notre modèle XGBoost de regression, cela parait être la méthode la plus "logique" (fariquants "mal classés", petrol et diesel "importants").   
                
                Les émissions de CO2 sont donc principalement liées à: 
                -	La présence d'un mode électriques et son autonomie. 				37 %
                -	Le poids du véhicule.								23 %
                -	Les caractéristiques moteur (cylindrée et puissance).				21 %
                -	La présence et l'efficacité de technologies innovantes de réduction de CO2.	4 %	
                 
                ## Les Shap_Values - Outils graphiques

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


        st.write('## Arbre de décision')
        st.write('##### XGB: Arbre de décision de rang 0, sur 4 niveau"')
        st.write("""
                    Il ne permet que d'avoir une représentation simpliste de l'algorithme utilisé par XGBoost.  
                    Dans la réalité pour notre modèle, il se combine avec 499 autres arbres de poids moindre, et sa profondeur va jusqu’à 6.""")
        
        img_name="xgb_plot_tree.png"
        st.image(images_path + img_name, use_column_width= True )

                 
  
                 


####################################### RELIQUAT A LAISSER ##############################
# Pour le moment (peut etre à utiliser)


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


  