from IPython.display import Image, display
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import colorlover as cl

import re
import os
import tempfile
import json
import datetime
from itertools import cycle

import math
import scipy.stats as stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import label_binarize, scale, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from auto_co2.styles import generate_styles
import numpy as np



########## Plotly Tools ##########

def show_if_interactive(fig, interactive):
    """Displays the Plotly figure interactively if interactive=True, 
    otherwise saves it to a temp PNG file and displays that."""
    if interactive:
        fig.show()
    else:
        with tempfile.NamedTemporaryFile(suffix='.png') as temp:
            fig.write_image(temp.name)
            display(Image(filename=temp.name))
        


def sample_data_if_needed(data, sample):
    """
    If sample is provided, returns a random sample of the data. 
    Otherwise returns the full data.
    
    Parameters:
    - data (DataFrame, Series, ndarray): The data to sample from
    - sample (int, float): If int, number of samples to return. If float, fraction of data to return.
    
    Returns:
    - DataFrame, Series, ndarray: Sampled data
    """
    if sample is not None:
        return sample_data(data, sample)
    return data


def sample_data(data, n):
    """
    Randomly samples data to the given size n. 
    
    If n is an int, samples n elements. 
    If n is a float, samples n * len(data) elements.
    
    Maintains pandas DataFrame/Series structure and numpy array structure.
    
    Parameters:
    - data (DataFrame, Series, ndarray): Data to sample from
    - n (int, float): Sample size
    
    Returns:
    Sampled data with same structure as input data 
    """
    if isinstance(n, float):
        n = int(n * len(data))
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.sample(n, random_state=42)
    elif isinstance(data, np.ndarray):
        np.random.seed(42)
        if len(data.shape) == 1:
            return np.random.choice(data, size=n, replace=False)
        elif len(data.shape) == 2:
            indices = np.random.choice(data.shape[0], size=n, replace=False)
            return data[indices]
    else:
        raise TypeError("Invalid data type. Must be a pandas DataFrame, Series, or a numpy array.")


def save_if_needed(fig, save, format=None):
    """Saves the figure to file if save is True.
    
    The figure is saved in the output/figures directory.
    
    Args:
        fig: The plotly figure to save 
        save (bool): If True, saves the figure
        format (str): The file format to save as. Valid options are 'png', 
            'html', and 'json'. Default 'png'.
    """
    if save:
        if format is None:
            save_plotly_fig(fig, 'png')
        else:
            save_plotly_fig(fig, format)
        


def save_plotly_fig(fig, format='png'):
    """
    Saves a Plotly figure to file in the specified format.
    
    Args:
        fig: The Plotly figure to save
        format (str): The file format to save as. Valid options are 
            'png', 'html', and 'json'. Default 'png'.
    """
    valid_formats = ['html', 'png', 'json']
    if format not in valid_formats:
        raise ValueError(f"Invalid format. Must be one of {valid_formats}")

    os.makedirs("../output/figures", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = fig.layout.title.text if fig.layout.title.text else 'untitled'
    filename = f"{base_filename}_{timestamp}"
    filename = re.sub('[^a-zA-Z0-9-_]', '_', filename)

    if format == 'html':
        fig.write_html(f"../output/figures/{filename}.html")
        print(f"Saved interactive figure to output/figures/{filename}.html")
    elif format == 'png':
        fig.write_image(f"../output/figures/{filename}.png")
        print(f"Saved static figure to output/figures/{filename}.png")
    elif format == 'json':
        fig.write_json(f"../output/figures/{filename}.json")  
        print(f"Saved JSON interactive figure to output/figures/{filename}.json")



def load_plotly_json(filename):
    """
    Loads a Plotly figure that was previously saved as a JSON file.
    
    Args:
        filename: The name of the JSON file containing the figure.
    
    Returns:
        A Plotly figure loaded from the provided JSON file.
    """

    output_dir = "../output/figures/"

    with open(f"{output_dir}{filename}", "r") as f:
        fig_dict = json.load(f)

    return go.Figure(fig_dict)


def add_legend(fig, text="Data Source: European Environment Agency, 2021"):
    """
    Adds a legend annotation to a Plotly figure.
    
    Args:
        fig (go.Figure): The Plotly figure to add the legend to.
        text (str, optional): The text for the legend. Defaults to 
            "Data Source: European Environment Agency, 2021".
    
    Returns:
        go.Figure: The updated figure with the added legend.
    """
    fig.update_layout(
        annotations=[
            dict(
                x=1,  # Rightmost position
                y=-0.15,  # Keep it slightly below the plot
                showarrow=False,
                text=text,
                xref="paper",
                yref="paper",
                xanchor='right',  # Anchor the text to the right
                yanchor='auto'
            )
        ]
    )
    return fig

def increase_font_size(fig, font_size=24):
    """Increase font size of text in Plotly figure.
    
    Args:
        fig (go.Figure): The Plotly figure 
        font_size (int): The font size to set for figure title.
                         Default is 24.
                         
    Returns:
        go.Figure: Updated figure with increased font size.
    """
    fig.update_layout(
        height=400, 
        width=1200, 
        title_font=dict(size=font_size)
    )
    return fig


########## End of Plotly Tools ##########

########## Countrywise visualization #########
def co2_emissions_viz(countrie, save=True, format='png'): # Plotly Express
    per_co2 = countries.data.sort_values(by='Co2EmissionsWltp', ascending=False)
    fig = px.bar(per_co2, x='Country', y='Co2EmissionsWltp', color='Co2EmissionsWltp', color_continuous_scale='Blues')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus polluants? (CO2)",'x': 0.3})
    fig.show()
    save_if_needed(fig, save, format)

        

def engine_power_viz(countries, save=True, format='png'): # Plotly Express
    per_engine = countries.data.sort_values(by='EnginePower', ascending=False)
    fig = px.bar(per_engine, x='Country', y='EnginePower', color='EnginePower', color_continuous_scale='Greens')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus puissants? (KWh)",'x': 0.3})
    fig.show()
    save_if_needed(fig, save, format)

        

def mass_viz(countries, save=True, format='png'): # Plotly Express
    per_mass = countries.data.sort_values(by='MassRunningOrder', ascending=False)
    fig = px.bar(per_mass, x='Country', y='MassRunningOrder', color='MassRunningOrder', color_continuous_scale='Reds')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus lourds? (Kg)",'x': 0.3})
    fig.show()
    save_if_needed(fig, save, format)

        

def countrywise_viz(countries, save=True, format='png'): # Plotly Express
        co2_emissions_viz(countries, save=save, format=format)
        engine_power_viz(countries, save=save, format=format)
        mass_viz(countries, save=save, format=format)


########## End of countrywise visualization #########


########## Manufacturerwise visualization #########
def plot_popular_fueltype(manufacturers, interactive=False, save=True, format='png'):
    """
    plot_popular_fueltype visualizes the distribution of fuel types by car make/manufacturer using a stacked bar chart.
    
    It groups the input DataFrame by Pool and Make columns, sums the Counts, 
    creates traces for each Make, assigns colors, and plots a stacked bar chart 
    with Make on the x-axis, sum of Counts on the y-axis, and fuel Pool split by color.
    
    The figure is displayed and can be saved to file optionally.
    """
    grouped_df = manufacturers.data.groupby(['Pool', 'Make'])['Count'].sum().reset_index(name='Counts')

    # Create a list of traces
    traces = []
    colors = cl.scales['12']['qual']['Paired']  # Get a list of 12 distinct colors
    make_colors = {make: colors[i % len(colors)] for i, make in enumerate(grouped_df['Make'].unique())}  # Create a color map
    
    for make in grouped_df['Make'].unique():
        df = grouped_df[grouped_df['Make'] == make]
        traces.append(go.Bar(x=df['Pool'], y=df['Counts'], name=make, marker_color=make_colors[make]))

    layout = go.Layout(barmode='stack', title="Répartition des types de carburants par marque/groupe automobile")
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)






########## End of manufacturerwise visualization #########




######### Context visualization #########

def plot_registrations_per_month(df:pd.DataFrame, interactive=True, save=True, format='png'):
    """
    """
    month_map = {1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 
                 5: 'Mai', 6: 'Juin', 7: 'Juil.', 8: 'Août', 
                 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Déc.'}

    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'])
    df['Month'] = df['RegistrationDate'].dt.month.map(month_map)  # Map month numbers to names
    monthly_counts = df.groupby('Month').size().reset_index(name='counts')

    fig = px.bar(monthly_counts, x='Month', y='counts', color='counts', color_continuous_scale='Purples')

    fig.update_layout(
        title={
            'text': "Nombre d'immatriculations par mois",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Mois",
        yaxis_title="Nombre d'immatriculations",
        showlegend=False,
        xaxis=dict(tickangle=-45)  # Tilt x-axis labels
    )
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)

########## End context visualization ##########






######### Case spectific visualization #########

def plot_fueltype_distribution(df:pd.DataFrame, interactive=True, save=True, format='png'):
    """
    Plot boxplots of CO2 emissions by fuel type.
    
    This function takes a DataFrame as input and generates a multi-boxplot 
    figure with one box per fuel type, showing the distribution of CO2
    emissions for that fuel type.
    
    Args:
        df (DataFrame): DataFrame containing FuelType and Co2EmissionsWltp columns
        interactive (bool): If True, show figure interactively. Default True.
        save (bool): If True, save figure to file. Default True. 
        format (str): File format if saving. Default 'png'.
    
    Returns:
        plotly.graph_objects.Figure: The generated multi-boxplot figure.
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
        hovermode='x',  # Limit the number of hover labels
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
    
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)





def plot_correlation_heatmap(df:pd.DataFrame, title='', interactive=True, save=True, format='png'):
    """
    Generates a heatmap visualization of the correlation matrix for numeric columns in a DataFrame.
    
    This function calculates the correlation matrix for the numeric columns in the input 
    DataFrame, then visualizes it as a heatmap using Plotly. Useful for exploring correlations
    between variables.
    
    Args:
      df (pd.DataFrame): DataFrame containing only numeric columns to correlate.
      title (str): Optional title for the plot.
      interactive (bool): If True, show the plot interactively.
      save (bool): If True, save the plot to a file.
      format (str): File format if saving, e.g. 'png'.
    
    Returns:
      plotly.graph_objects.Figure: The Plotly figure object for the correlation heatmap.
    
    """
    df = df.select_dtypes(include=[np.number])
    correlation_matrix = df.corr()
    heatmap = go.Heatmap(z=correlation_matrix, 
                         x=correlation_matrix.columns, 
                         y=correlation_matrix.columns, 
                         colorscale='RdBu_r')

    # Affichage des coefficients de corrélation
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            color  = 'white' if value < -0.5 or value > 0.5 else 'black'
            annotations.append(
                go.layout.Annotation(text=str(round(value, 2)), 
                                                    x=correlation_matrix.columns[j], 
                                                    y=correlation_matrix.columns[i], 
                                                    showarrow=False, 
                                                    font=dict(color=color, size=16)))

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=dict(
            text=f'Corrélation des variables quantitatives {title}',
            x=0.55,  # Center the title
            font=dict(size=24)),
        autosize=False,
        width=1000,
        height=850,
        annotations=annotations,
        xaxis=dict(autorange='reversed', tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)))

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)



def plot_qqplots(df:pd.DataFrame, title='', interactive=True, save=True, format='png'):
    """
    Plot QQ plots for numeric columns in a DataFrame.
    
    This function samples the DataFrame, calculates theoretical and sample quantiles for
    numeric columns, and creates a Plotly figure with QQ plots comparing them. Useful for 
    visually checking if the data matches a normal distribution.
    
    Args:
      df (pd.DataFrame): DataFrame containing numeric columns to plot.
      title (str): Plot title.
      interactive (bool): If True, show plot interactively.
      save (bool): If True, save plot to file.
      format (str): File format if saving, e.g. 'png'.
    
    Returns:
      plotly.graph_objects.Figure: Plotly figure object with QQ plots.
    """
    n_samples = min(10000, df.shape[0])  # Sample 10,000 or the total number of rows, whichever is smaller
    df_sample = df.select_dtypes(include=[np.number]).sample(n=n_samples, random_state=1)

    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(rows=4, cols=2)

    # Loop over the first 4 columns of the DataFrame
    for i, col in enumerate(df_sample.columns[0:8]):
        # Calculate the theoretical quantiles and order them
        theoretical_quantiles = np.sort(stats.norm.ppf((np.arange(len(df_sample[col])) + 0.5) / len(df_sample[col])))
        
        # Calculate the sample quantiles and order them
        sample_quantiles = np.sort(df_sample[col])
        
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name=col), row=(i//2)+1, col=(i%2)+1)

    # Update layout
    fig.update_layout(height=1000, width=800, title_text=f"Quantiles observés vs quantiles d'une distribution normale (QQ Plots) {title}")

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)



def plot_feature_distributions(df, title='', interactive=True, save=True, format='png'):
    """
    Plot histograms of numeric feature distributions.
    
    For each numeric column in the input DataFrame, calculate a histogram 
    with 40 bins and add it as a subplot to a Plotly figure. Configure the
    layout for the number of rows and columns needed to fit all histograms.
    
    Args:
      df (DataFrame): Input DataFrame.
      title (str): Plot title.
      interactive (bool): If True, show plot interactively.
      save (bool): If True, save plot to file.
      format (str): File format if saving, e.g. 'png'.
    
    Returns:
      plotly.graph_objects.Figure: Plotly figure with histograms.
    """
    df_num = df.select_dtypes(include=[np.number])
    cols = [col for col in df_num.columns]  # Only keep numeric columns

    # Calculate the number of rows and columns for the subplot grid
    n_cols = 3
    n_rows = -(-len(cols) // n_cols)  # Ceiling division

    fig = sp.make_subplots(rows=n_rows, cols=n_cols)

    for i, col_name in enumerate(cols):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(go.Histogram(x=df[col_name], nbinsx=40, name=col_name), row=row, col=col)
        fig.update_xaxes(title_text=col_name, row=row, col=col)  # Add x-axis label

    fig.update_layout(height=1000, width=1100, title_text=f"Distribution des variables explicatives {title}", showlegend=False)  

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        
        
        
def plot_continuous_distribution(s, title='', interactive=True, save=True, format='png'):
    """
    Plot the kernel density estimate and histogram for a continuous variable.
    
    This function takes a pandas Series, calculates the kernel density estimate
    using scipy.stats.gaussian_kde and plots it along with a histogram of the data.
    It configures the plot layout including axis labels, titles, legend and 
    transparency.
    
    The figure is displayed interactively if interactive=True, and saved to file
    if save=True.
    """
    # Calculate the Kernel Density Estimation of the series
    x = np.linspace(s.min(), s.max(), 1000)
    kde = gaussian_kde(s)
    y = kde.evaluate(x)

    # Create the line plot for the KDE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='KDE', yaxis='y2'))

    # Add a histogram
    fig.add_trace(go.Histogram(x=s, nbinsx=40, name='Histogram'))

    # Add title and labels
    fig.update_layout(
        title={
            'text': f'Courbe KDE de {title}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=20
            )
        },
        xaxis_title=s.name,
        yaxis_title='Count',
        yaxis2=dict(
            title='Density',
            overlaying='y',
            side='right'
        ),
        barmode='overlay'
    )

    # Make the histogram semi-transparent
    fig.data[1].marker.line.width = 1
    fig.data[1].marker.line.color = "black"
    fig.data[1].opacity = 0.5

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        
######### End of case specific visualization #########


########## Model Visualizations ##########


def plot_confusion_matrix(y_true, y_pred, 
                          palette='Blues', 
                          classes=None, 
                          interactive=True, 
                          save=True,
                          format='png', 
                          title=''):
    """
    """
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    heatmap = go.Heatmap(z=cm, 
                         x=classes, 
                         y=classes, 
                         colorscale=palette)

    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            color  = 'white' if value > cm.max() / 2 else 'black'
            annotations.append(
                go.layout.Annotation(text=str(value), 
                                     x=j, 
                                     y=i, 
                                     showarrow=False, 
                                     font=dict(color=color, size=16)))

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=dict(
            text=f"Matrice de confusion {title}",
            x=0.5,  # Center the title
            font=dict(size=24)),
        autosize=False,
        width=600,
        height=600,
        annotations=annotations,
        xaxis=dict(title='Labels prédits', tickfont=dict(size=14), tickvals=list(range(len(classes))), ticktext=classes),
        yaxis=dict(title='Vrais labels', tickfont=dict(size=14), tickvals=list(range(len(classes))), ticktext=classes))

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        




def plot_actual_vs_predicted(y_true, y_pred, interactive=False, save=True, sample=None):
    y_true = sample_data_if_needed(y_true, sample)
    y_pred = sample_data_if_needed(y_pred, sample)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', marker=dict(color='#2E8B57', size=5, opacity=0.5), name='Predicted'))
    fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', line=dict(color='red'), name='Actual'))
    fig.update_layout(title='Actual vs Predicted')
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, 'png')
    return fig

def plot_residuals_distribution(residuals, interactive=False, save=True, sample=None):
    residuals = sample_data_if_needed(residuals, sample)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40, name='Residuals'))
    fig.update_layout(title='Residuals Distribution')
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, 'png')
    return fig

def plot_residuals_vs_fitted(y_true, y_pred, interactive=True, save=False, format='png'):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    fig.update_layout(
        title='Residuals vs Fitted',
        xaxis_title='Fitted values',
        yaxis_title='Residuals',
        autosize=False,
        width=500,
        height=500
    )
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
    return fig

def plot_qq_plot(residuals, interactive=False, save=True, sample=None):
    residuals = sample_data_if_needed(residuals, sample)
    
    fig = go.Figure()
    residuals_norm = scale(residuals)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals_norm, dist='norm', fit=True)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Observed'))
    fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Theoretical'))
    fig.update_layout(
        title='QQ Plot',
        autosize=False,
        width=500,
        height=500,
        xaxis=dict(title="Theoretical quantiles", scaleanchor='y'),
        yaxis=dict(title="Observed quantiles", scaleanchor='x')
    )
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, 'png')
    return fig




def plot_regression_diagnostics(y_true, y_pred, residuals, title, interactive=True, save=True, format='png', sample=None):
    """
    Plot regression diagnostics.
    
    Plots 4 subplots:
    1) Actual vs predicted values
    2) Residuals distribution 
    3) Residuals vs fitted
    4) QQ plot of residuals
    
    Args:
      y_true: Array of true target values
      y_pred: Array of predicted target values
      residuals: Array of residual values
      title: Plot title 
      interactive: Whether to display plot interactively
      save: Whether to save plot to file
      format: File format if saving 
      sample: Optionally sample a subset of the data
      
    Returns:
      Figure object with regression diagnostic plots 
    """
    y_true = sample_data_if_needed(y_true, sample)
    y_pred = sample_data_if_needed(y_pred, sample)
    residuals = sample_data_if_needed(residuals, sample)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Actual vs Predicted",
                                                       "Residuals Distribution",
                                                       "Residuals vs Fitted",
                                                       "Residuals QQ Plot (x: Theoretical, y: Observed)"))

    # Actual vs Predicted values
    fig1 = plot_actual_vs_predicted(y_true, y_pred, interactive=False, save=False)
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig1.data[1], row=1, col=1)

    # Residuals Distribution
    fig2 = plot_residuals_distribution(residuals, interactive=False, save=False)
    fig.add_trace(fig2.data[0], row=1, col=2)

    # Residuals versus fitted
    fig3 = plot_residuals_vs_fitted(y_true, y_pred, interactive=False, save=False)
    fig.add_trace(fig3.data[0], row=2, col=1)

    # QQ Plot
    fig4 = plot_qq_plot(residuals, interactive=False, save=False)
    fig.add_trace(fig4.data[0], row=2, col=2)
    fig.add_trace(fig4.data[1], row=2, col=2)

    fig.update_layout(height=700, width=900, title_text=title)
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save)


    
        
        
def plot_lr_elasticnet(lr_en, title='', interactive=True, save=True, format='png', sample=None):
    """
    """
    alphas = lr_en.alphas_

    fig = go.Figure()
    for i in range(lr_en.mse_path_.shape[0]):
        mse_path_sample = sample_data_if_needed(lr_en.mse_path_[i,:,:], sample)
        fig.add_trace(go.Scatter(x=alphas, y=mse_path_sample.mean(axis=1), mode='lines', name='Moyenne pour l1_ratio= %.2f' %lr_en.l1_ratio[i]))

    fig.update_layout(title='Mean squared error pour chaque lambda'+title, xaxis_title='Alpha', yaxis_title='Mean squared error')

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
    
    
        
def plot_mse_folds(lr_en, l1_ratios, interactive=True, save=True, format='png'):
    """
    Plot mean squared error vs alpha for elastic net model.
    
    For each l1_ratio, create subplots showing MSE across alpha values for each fold.
    Also plot average MSE across folds. Indicate chosen alpha with vertical line."""
    # Define a color for each fold number
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Calculate the number of rows and columns for the subplots
    n_rows = math.ceil(len(l1_ratios) / 2)
    n_cols = 2 if len(l1_ratios) > 1 else 1

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'l1_ratio={l1_ratio}' for l1_ratio in l1_ratios])

    # Add a line for each fold in each subplot
    for i, l1_ratio in enumerate(l1_ratios):
        row = i // n_cols + 1
        col = i % n_cols + 1
        for j, fold in enumerate(lr_en.mse_path_[i]):
            fig.add_trace(go.Scatter(x=lr_en.alphas_, y=fold, mode='lines', name=f'Fold {j+1}', line=dict(color=colors[j])), row=row, col=col)

        # Add a line for the average MSE across folds in each subplot
        avg_mse = lr_en.mse_path_[i].mean(axis=0)
        fig.add_trace(go.Scatter(x=lr_en.alphas_, y=avg_mse, mode='lines', name='Average across the folds', line=dict(color='black', width=2, dash='dot')), row=row, col=col)

        # Add a vertical line for the chosen alpha in each subplot
        fig.add_shape(type='line', x0=lr_en.alpha_, x1=lr_en.alpha_, y0=avg_mse.min(), y1=avg_mse.max(), line=dict(color='black', dash='dash'), row=row, col=col)

    # Update y-axes to have the same scale
    fig.update_yaxes(matches='y')
    fig.update_layout(height=300*n_rows, width=600*n_cols, title_text="Mean squared error pour chaque lambda")
    fig.update_xaxes(title_text='Alphas')
    
    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)


def plot_pca_variance(pca, n_features,  title='', interactive=True, save=True, format='png'):
    """
    Plot the explained variance ratio of the principal components.

    This function takes a PCA object and the number of features to plot. 
    It generates a plot with two lines:
    - Variance: Shows the explained variance ratio per principal component
    - Cumulative Variance: Shows the cumulative explained variance ratio

    It also updates the plot layout, including the titles and axis labels.
    """
    x_values = list(range(1, n_features+1))

    # Create a scatter plot of the explained variance ratio
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=pca.explained_variance_ratio_[:n_features], mode='lines+markers', name='Variance'))
    fig.add_trace(go.Scatter(x=x_values, y=np.cumsum(pca.explained_variance_ratio_)[:n_features], mode='lines+markers', name='Cumulative Variance'))

    fig.update_layout(title='Part de la variance expliquée, composants PCA'+title,
                      xaxis=dict(title='Component'),
                      yaxis=dict(title='Part de la variance expliquée'))
    fig.update_layout(height=700, width=1000)


    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        
        
        
def plot_xgboost(results, metric='mlogloss', title='', interactive=True, save=True, format='png'):
    """
    Plot XGBoost results.
    
    This function takes the results dictionary from XGBoost training and plots
    the given metric over epochs for each model in the results.
    
    Args:
      results: Dictionary of model results from XGBoost, where keys are model
              names and values are dictionaries of metrics over epochs.
      metric: Metric to plot on y-axis, default is 'mlogloss'.
      title: Plot title.
      interactive: If True, show plot interactively.
      save: If True, save plot to file.
      format: File format if saving, default 'png'.
    
    Returns:
      None.
    """
    num_epochs = len(next(iter(results.values()))[metric])
    x_axis = list(range(0, num_epochs))

    fig = go.Figure()
    for key in results.keys():
        fig.add_trace(go.Scatter(x=x_axis, y=results[key][metric], mode='lines', name=key))

    fig.update_layout(title='XGBoost'+title, xaxis_title='Epoch', yaxis_title=metric)

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)   
    

def plot_roc_curves(y_test, y_pred, title='', interactive=True, save=True, format='png'):
        """
        Plot ROC curves.
        
        For each class, compute the ROC curve and ROC area using sklearn's 
        roc_curve and auc functions. Create a plotly figure with one trace per 
        class, plus a diagonal reference line. Update the layout for titles, axes,
        size. Show or save the figure based on input args.
        
        Args:
            y_test: Array of truth labels.
            y_pred: Array of predicted probabilities.
            title: Plot title.
            interactive: If True, show plot interactively.
            save: If True, save plot.
            format: File format if saving.
            
        Returns:
            None.
        """
        # Determine unique classes
        classes = np.unique(y_test)

        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = roc_curve(y_test_bin.ravel(), y_pred.ravel())

        # Create a Figure
        fig = go.Figure()

        # Add ROC curves to the Figure
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue', 'yellow'])
        for i, color in zip(range(n_classes), colors):
                fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]), line=dict(color=color)))

        # Add diagonal line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='black', dash='dash')))

        # Update layout
        fig.update_layout(title='Courbes ROC: '+title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', autosize=False, width=600, height=600, margin=dict(l=50, r=50, b=100, t=100, pad=4))
        fig.update_layout(height=700, width=1000)

        show_if_interactive(fig, interactive)
        save_if_needed(fig, save, format)
        
        

def plot_feature_importance(model, max_num_features=20, title='', interactive=True, save=True, format='png'):
    """
    Plot feature importance.
    
    Plots a horizontal bar chart showing the importance of each feature in the model. 
    The feature importances are extracted from the model and sorted. The max_num_features
    argument limits the number of features plotted.
    
    Args:
        model: The trained model. Must have a get_booster() method that returns feature importances.
        max_num_features: The maximum number of features to plot.
        title: The plot title. 
        interactive: If True, show the plot interactively.
        save: If True, save the plot.
        format: The image format if saving the plot.
        
    Returns:
        None.
    """
    
    importance = model.get_booster().get_score(importance_type='weight')
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=False))
    limited_importance = dict(list(sorted_importance.items())[:max_num_features])

    fig = go.Figure()
    for key in limited_importance.keys():
        fig.add_trace(go.Bar(y=[key], x=[limited_importance[key]], orientation='h', name=key))

    fig.update_layout(title='Feature Importance'+title, yaxis_title='Features', xaxis_title='Importance')
    fig.update_layout(height=700, width=1000)

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        
        
        
def plot_training_history(training_history, title='', interactive=True, save=True, format='png'):
    """
    Plot the training history.
    
    Plots the training and validation accuracy and loss from the training history 
    of a Keras model over epochs. Useful for visualizing how well the model trained.
    
    Args:
      training_history: The Keras training History object containing the history of
                        metrics from training.
      title: An optional title for the plot.
      interactive: If True, show the plot interactively.
      save: If True, save the plot.
      format: The image format if saving the plot.
      
    Returns:
      None.
    """
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Scatter(y=training_history.history['accuracy'], mode='lines', name='Train Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=training_history.history['val_accuracy'], mode='lines', name='Validation Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=training_history.history['loss'], mode='lines', name='Train Loss'), row=2, col=1)
    fig.add_trace(go.Scatter(y=training_history.history['val_loss'], mode='lines', name='Validation Loss'), row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    fig.update_layout(height=600, width=600, title_text="Model Accuracy and Loss" + title)

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        
        

def plot_shap_summary(shap_explanation, feature_names, max_num_features=20, title='', interactive=True, save=True, format='png', sample=None):
    """Plot a summary of SHAP feature importance values.
    
    Calculates the mean absolute SHAP value per feature and plots a horizontal
    bar chart showing the most important features. Useful for understanding which
    features in the model have the most impact on predictions.
    """
    shap_values = shap_explanation.values

    shap_values = sample_data_if_needed(shap_values, sample)

    # Calculate the mean absolute SHAP values for each feature
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    importance = dict(zip(feature_names, mean_abs_shap_values))
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=False))

    # Limit the number of features
    limited_importance = dict(list(sorted_importance.items())[:max_num_features])

    fig = go.Figure()
    for key in limited_importance.keys():
        fig.add_trace(go.Bar(y=[key], x=[limited_importance[key]], orientation='h', name=key))

    fig.update_layout(title='SHAP Summary'+title, yaxis_title='Features', xaxis_title='Importance')

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)
        
        
        
def plot_shap_dependence(feature, interaction_feature, shap_values, X, title='', interactive=False, save=True, format='png', sample=None):
    """
    Plot the SHAP dependence between a feature and the SHAP values.
    
    Creates a 2D density plot showing the relationship between the given feature
    values and the corresponding SHAP values. Also adds a scatter plot colored 
    by the interaction feature values. Useful for understanding how the model
    depends on the specified feature.
    """
    # Sample the data if needed, shap is computationally expensive
    shap_values = sample_data_if_needed(shap_values, sample)
    X = sample_data_if_needed(X, sample)

    # Get the SHAP values and feature values for the specified features
    shap_values_feature = shap_values[:, X.columns.get_loc(feature)]
    feature_values = X[feature].values
    interaction_values = X[interaction_feature].values

    # Create a 2D density plot
    fig = ff.create_2d_density(
        x=feature_values, 
        y=shap_values_feature, 
        colorscale='Viridis',
        hist_color='rgba(0, 0, 0, 0)',
        point_size=3
    )

    # Add a scatter plot for the interaction feature
    fig.add_trace(
        go.Scatter(
            x=interaction_values, 
            y=shap_values_feature, 
            mode='markers',
            marker=dict(
                size=4,
                color=interaction_values,
                colorscale='Viridis',
                showscale=True
            ),
            text=interaction_values,
            name=interaction_feature
        )
    )

    fig.update_layout(
        title=f'SHAP Dependence Plot: {feature} vs SHAP values',
        xaxis_title=feature,
        yaxis_title='SHAP values',
        showlegend=True)

    show_if_interactive(fig, interactive)
    save_if_needed(fig, save, format)