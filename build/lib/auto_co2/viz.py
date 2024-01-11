from IPython.display import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
import io

from auto_co2.agg import Countries
from auto_co2.styles import generate_styles




########## Countrywise visualization #########
def co2_emissions_viz(countries: Countries): # Plotly Express
    per_co2 = countries.data.sort_values(by='Co2EmissionsWltp', ascending=False)
    fig = px.bar(per_co2, x='Country', y='Co2EmissionsWltp', color='Co2EmissionsWltp', color_continuous_scale='Blues')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus polluants? (CO2)",'x': 0.3})
    fig.show()

def engine_power_viz(countries: Countries): # Plotly Express
    per_engine = countries.data.sort_values(by='EnginePower', ascending=False)
    fig = px.bar(per_engine, x='Country', y='EnginePower', color='EnginePower', color_continuous_scale='Greens')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus puissants? (KWh)",'x': 0.3})
    fig.show()

def mass_viz(countries: Countries): # Plotly Express
    per_mass = countries.data.sort_values(by='MassRunningOrder', ascending=False)
    fig = px.bar(per_mass, x='Country', y='MassRunningOrder', color='MassRunningOrder', color_continuous_scale='Reds')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus lourds? (Kg)",'x': 0.3})
    fig.show()

def countrywise_viz(countries: Countries): # Plotly Express
    co2_emissions_viz(countries)
    engine_power_viz(countries)
    mass_viz(countries)


########## End of countrywise visualization #########



########## Manufacturerwise visualization #########










########## End of manufacturerwise visualization #########




######### Context visualization #########

def plot_registrations_per_month(df:pd.DataFrame):
    mois = ['Janvier', 'Février', 'Mars', 'Avril', 
                        'Mai', 'Juin', 'Juil.', 'Août', 
                        'Sept.', 'Oct.', 'Nov.', 'Déc.']


    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'])
    df['Month'] = df['RegistrationDate'].dt.month
    # Convert the groupby object to a DataFrame
    monthly_counts = df.groupby('Month').size().reset_index(name='counts')


    # Create a bar chart
    fig = px.bar(monthly_counts, x='Month', y='counts', color='counts', color_continuous_scale='Purples')

    # Update the layout
    fig.update_layout(
        title={
            'text': "Nombre d'immatriculations par mois",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Mois",
        yaxis_title="Nombre d'immatriculations",
        showlegend=False
    )

    # Update the x-axis tick labels with French month names
    fig.update_xaxes(ticktext=mois, tickvals=list(range(1, 13)))

    fig.show()

########## End context visualization ##########






######### Case spectific visualization #########

def plot_fueltype_distribution(df:pd.DataFrame, interactive=True):
    """Affiche les boites à moustaches des émissions de CO2 par FuelType

    Args:
        df (DataFrame): _description_
        interactive (bool, optional): True par défaut, False pour de meilleures performances.

    Returns:
        _type_: plolty multi-boxplot figure
    """

    fig = go.Figure()
    for fuel_type in df['FuelType'].unique():
        fig.add_trace(go.Box(
            x=df[df['FuelType'] == fuel_type]['Co2EmissionsWltp'],
            name=fuel_type,
            boxpoints='outliers'))
        
    fig.update_layout(
        yaxis=dict(
            title='FuelType',
            tickangle=45),
        
        xaxis=dict(
            title='Co2EmissionsWltp'),
        
        autosize=False,
        width=1400,
        height=800,
        annotations=[
            dict(
                x=0,
                y=0,
                showarrow=False,
                text="Data Source: European Environment Agency, 2021",
                xref="paper",
                yref="paper",
                font=dict(size=12))
        ]
    )
    
    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format='png')
        display(Image(img_bytes))














def plot_heatmap(df:pd.DataFrame, interactive=True):
    """Heatmap de corrélation des variables quantitatives

    Args:
        df (pd.DataFrame): Colonnes numériques uniquement
        interactive (bool, optional): True par défaut, False pour de meilleures performances.

    Returns:
        _type_: plotly heatmap figure
    """

    df = df.select_dtypes(include=[np.number])
    # Matrice de corrélation plotly
    correlation_matrix = df.corr()
    heatmap = go.Heatmap(z=correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns, colorscale='RdBu_r')

    # Affichage des coefficients de corrélation
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            color  = 'white' if value < -0.5 or value > 0.5 else 'black'
            annotations.append(go.layout.Annotation(text=str(round(value, 2)), 
                                                    x=correlation_matrix.columns[j], 
                                                    y=correlation_matrix.columns[i], 
                                                    showarrow=False, 
                                                    font=dict(color=color, size=16)))


    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=dict(
            text='Corrélation des variables quantitatives',
            x=0.55,  # Center the title
            font=dict(
                size=24  # Increase font size here
            )
        ),
        autosize=False,
        width=1000,
        height=850,
        annotations=annotations,
        xaxis=dict(
            autorange='reversed',
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            tickfont=dict(size=14)
        )
    )

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format='png')
        display(Image(img_bytes))
    return fig