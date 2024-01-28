import numpy as np
import pandas as pd
import requests
import io
from functools import partial
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import colorlover as cl

from auto_co2.styles import generate_styles, displayer
from auto_co2.viz import add_legend, save_plotly_fig, increase_font_size, save_if_needed


########## Toolkit ##########

def float_rounder(df, n=3):
        float_cols = df.select_dtypes(include='float64').columns
        return df[float_cols].round(n)

def rename_dict_keys(d:dict, old_keys:list, new_keys:list):
    if len(old_keys) != len(new_keys):
        raise ValueError("Both key lists must have the same length")
    for old_key, new_key in zip(old_keys, new_keys):
        d[new_key] = d.pop(old_key)
        
def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)     

def calculate_market_share(x):
    return x.size / x.size.sum() * 100  
    
########## End of toolkit ##########



########## Classes ##########
class CountryDataAggregator:
    """
    """
    ########## Default Class attributes ##########
    COUNTRY_STATS_QUERY_202301 = {
        # From wikipedia query
        'IE': {'population': 5123536, 'gdp_per_capita': 103296.79936336935},
        'HU': {'population': 9603634, 'gdp_per_capita': 18935.334502543515},
        'ES': {'population': 47415750, 'gdp_per_capita': 30103.513733200467},
        'BE': {'population': 11584008, 'gdp_per_capita': 51785.18523122567},
        'LU': {'population': 660809, 'gdp_per_capita': 129396.30639715864},
        'FI': {'population': 5563970, 'gdp_per_capita': 53269.09118201572},
        'SE': {'population': 10548336, 'gdp_per_capita': 60375.042698298574},
        'DK': {'population': 5827463, 'gdp_per_capita': 68349.34391930074},
        'PL': {'population': 38382576, 'gdp_per_capita': 17929.40124589345},
        'LT': {'population': 2860002, 'gdp_per_capita': 24592.39504308039},
        'IT': {'population': 58850717, 'gdp_per_capita': 35927.442598770715},
        'AT': {'population': 8979894, 'gdp_per_capita': 53493.772186286384},
        'GR': {'population': 10482487, 'gdp_per_capita': 20898.272753975274},
        'PT': {'population': 10347892, 'gdp_per_capita': 24544.40455804912},
        'FR': {'population': 67749632, 'gdp_per_capita': 43658.97897812936},
        'GB': {'population': 67326569, 'gdp_per_capita': 46378.1296493068},
        'DE': {'population': 83149300, 'gdp_per_capita': 51232.36048676297},
        'EE': {'population': 1373101, 'gdp_per_capita': 27748.00466899376},
        'LV': {'population': 1883008, 'gdp_per_capita': 21855.4104193928},
        'CZ': {'population': 10827529, 'gdp_per_capita': 23152.189202171612},
        'SK': {'population': 5449270, 'gdp_per_capita': 21774.768346769382},
        'SI': {'population': 2066880, 'gdp_per_capita': 30053.882187161325},
        'RO': {'population': 19053815, 'gdp_per_capita': 15811.089953586723},
        'BG': {'population': 7000039, 'gdp_per_capita': 12719.986046649168},
        'HR': {'population': 3871833, 'gdp_per_capita': 18328.426475263783},
        'CY': {'population': 1141166, 'gdp_per_capita': 24921.04806925548},
        'MT': {'population': 465292, 'gdp_per_capita': 38180.90578604403},
        'NL': {'population': 17100715, 'gdp_per_capita': 48569.46734975701}}
    
    MAIN_AGGREGATOR = 'Country'
    COUNT_COLUMN = 'Count'
    REQUIRED_COLUMNS = ['Country', 'Make', 'FuelType', 'FuelConsumption', 
                        'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']
    DROPNA_SUBSET = ['FuelConsumption']
    AGGREGATION_DICT = {
        'ID': 'count',
        'Make': 'first',
        'FuelType': lambda x: x.mode()[0],
        'FuelConsumption': 'mean',
        'EnginePower': 'mean',
        'Co2EmissionsWltp': 'mean',
        'MassRunningOrder': 'mean',
        'BaseWheel': 'mean',
        'AxleWidthSteering': 'mean',
        'ElectricRange': 'mean'}
    RENAME_COLUMNS = {'ID': 'Count'}
    DISPLAY_STYLES = generate_styles()
    
    CO2_EMISSIONS_VIZ_PARAMS = {
        "x": "Country",
        "y": "Co2EmissionsWltp",
        "color": "Co2EmissionsWltp",
        "color_continuous_scale": "Blues",
        "title": ["Quel pays achète les véhicules les plus polluants? (CO2)", "Quel pays achète les véhicules les moins polluants? (CO2)"],
    }

    ENGINE_POWER_VIZ_PARAMS = {
        "x": "Country",
        "y": "EnginePower",
        "color": "EnginePower",
        "color_continuous_scale": "Greens",
        "title": ["Quel pays achète les véhicules les plus puissants? (KWh)", "Quel pays achète les véhicules les moins puissants? (KWh)"],
    }

    MASS_VIZ_PARAMS = {
        "x": "Country",
        "y": "MassRunningOrder",
        "color": "MassRunningOrder",
        "color_continuous_scale": "Reds",
        "title": ["Quel pays achète les véhicules les plus lourds? (Kg)", "Quel pays achète les véhicules les moins lourds? (Kg)"],
    }
    ########## End of Default Class attributes ##########
    
    def __init__(self, df:pd.DataFrame, country_stats: dict=None, required_columns=None, 
                 aggregation_dict=None, rename_columns=None, dropna=True, dropna_subset=None):
        """
        Initializes the CountryDataAggregator object.

        Args:
            df (pd.DataFrame): The input DataFrame containing car data.
            country_stats (dict, optional): Additional country statistics. Defaults to None.
            required_columns (list, optional): List of required columns in the DataFrame. Defaults to None.
            aggregation_dict (dict, optional): Dictionary specifying the aggregation functions for each column. Defaults to None.
            rename_columns (dict, optional): Dictionary specifying the new column names. Defaults to None.

        Raises:
            ValueError: If the DataFrame does not contain all the required columns.
        """
        if required_columns is None:
            required_columns = self.REQUIRED_COLUMNS
        if dropna:
            if dropna_subset is None:
                dropna_subset = self.DROPNA_SUBSET
            df.dropna(subset=dropna_subset, inplace=True)
        if aggregation_dict is None:
            aggregation_dict = self.AGGREGATION_DICT
        if rename_columns is None:
            rename_columns = self.RENAME_COLUMNS

        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        pd.set_option('display.float_format', '{:.3f}'.format)

        self._df = df.groupby(self.MAIN_AGGREGATOR).agg(aggregation_dict)
        self._df = self._df.rename(columns=rename_columns)
        
        country_stats = pd.DataFrame(self.COUNTRY_STATS_QUERY_202301).transpose()
        country_stats.index.name = self.MAIN_AGGREGATOR
        self._df = self._df.join(country_stats, how='inner')

        gdp = self._df.pop('gdp_per_capita')
        self._df.insert(0, 'GdpPerCapita', gdp)
        
        pop = self._df.pop('population')        
        self._df.insert(0, 'Population', pop)

        self._df.reset_index(inplace=True)
                
    def __repr__(self):
        """
        Returns a string representation of the CountryDataAggregator object.
        """
        return f"Countries(data={self._df.head()})"
    
    def sort(self, by=None, ascending=False):
        """
        Sorts the data by a specified column.

        Args:
            by (str): The column to sort by. Defaults to COUNT_COLUMN.
            ascending (bool): Whether to sort in ascending order. Defaults to False.
        """
        if by is None:
            by = self.COUNT_COLUMN
        self._df.sort_values(by=by, ascending=ascending, inplace=True)

    def display(self, n=None, styles=None, title=None, save=True):
        """
        Displays the data with optional styling and title.

        Args:
            n (int, optional): The number of rows to display. Defaults to None.
            styles (dict, optional): The styles to apply to the data. Defaults to None.
            title (str, optional): The title of the display. Defaults to None.
        """
        if styles is None:
            styles = generate_styles()
        displayer(self._df, n, styles, title, save=save)
    
    def display_sorted(self, by=None, ascending=False, n=None, styles=None, title=None, columns=None, save=True):
        """
        Sorts the data and displays it with optional styling and title.

        Args:
            by (str, optional): The column to sort by. Defaults to COUNT_COLUMN.
            ascending (bool, optional): Whether to sort in ascending order. Defaults to False.
            n (int, optional): The number of rows to display. Defaults to None.
            styles (dict, optional): The styles to apply to the display. Defaults to None.
            title (str, optional): The title of the display. Defaults to None.
            columns (list, optional): The columns to display. Defaults to None.
        """
        if by is None:
            by = self.COUNT_COLUMN
        self.sort(by=by, ascending=ascending)

        # If columns are specified, select only those columns
        if columns is not None:
            self._df = self._df[columns]
        displayer(self._df, n, styles, title, save=save)

        
    def co2_emissions_viz(self, ascending=False, params=None, save=True, format='png'):
        if params is None:
            params = self.CO2_EMISSIONS_VIZ_PARAMS
        per_co2 = self._df.sort_values(by=params["y"], ascending=ascending)
        params["title"] = params["title"][0] if ascending is False else params["title"][1]
        fig = px.bar(per_co2, **params)
        fig = add_legend(fig)
        fig.show()
        save_if_needed(fig, save, format)

    def engine_power_viz(self, ascending=False, params=None, save=True, format='png'):
        if params is None:
            params = self.ENGINE_POWER_VIZ_PARAMS
        per_engine = self._df.sort_values(by=params["y"], ascending=ascending)
        params["title"] = params["title"][0] if ascending is False else params["title"][1]
        fig = px.bar(per_engine, **params)
        fig = add_legend(fig)
        fig.show()
        save_if_needed(fig, save, format)

    def mass_viz(self, ascending=False, params=None, save=True, format='png'):
        if params is None:
            params = self.MASS_VIZ_PARAMS
        per_mass = self._df.sort_values(by=params["y"], ascending=ascending)
        params["title"] = params["title"][0] if ascending is False else params["title"][1]
        fig = px.bar(per_mass, **params)
        fig = add_legend(fig)
        fig.show()
        save_if_needed(fig, save, format)            

    def countrywise_viz(self, save=True, format='png'):
        self.co2_emissions_viz(False, save=save, format=format)
        self.engine_power_viz(False, save=save, format=format)
        self.mass_viz(False, save=save, format=format)


class ManufacturerDataAggregator:
    """
    Class for aggregating and analyzing manufacturer data.

    Attributes:
        MAIN_AGGREGATOR (str): The main aggregator column name.
        SECONDARY_AGGREGATOR (str): The secondary aggregator column name.
        GROUPBY_COLUMNS (list): List of columns to group by.
        COUNT_COLUMN (str): The count column name.
        REQUIRED_COLUMNS (list): List of required columns in the DataFrame.
        AGGREGATION_DICT (dict): Dictionary specifying the aggregation functions for each column.
        RENAME_COLUMNS (dict): Dictionary specifying the column renaming.
        DISPLAY_STYLES (dict): Dictionary specifying the display styles.
        MASS_ENGINE_SCATTER_PARAMS (dict): Dictionary specifying the parameters for the mass-engine scatter plot.
        FACETPLOT_PARAMS (dict): Dictionary specifying the parameters for the facet plot.

    Methods:
        __init__(self, df: pd.DataFrame, required_columns=None, aggregation_dict=None,
                 groupby_columns=None, rename_columns=None): Initializes the ManufacturerDataAggregator object.
        __repr__(self): Returns a string representation of the ManufacturerDataAggregator object.
        sort(self, by=self.COUNT_COLUMN, ascending=False): Sorts the data by a specified column.
        display(self, n=None, styles=None, title=None): Displays the data with specified styles and title.
        display_sorted(self, by=COUNT_COLUMN, ascending=False, n=None, styles=None, title=None): Sorts and displays the data.
        plot_popular_fueltype(self, save=True, format='png'): Plots the popular fuel types by manufacturer.
        plot_scatter(self, df, plot_params): Plots a scatter plot with specified parameters.
        plot_mass_engine_scatter(self): Plots the mass-engine scatter plot.
        facetplot(self, plot_params): Plots a facet plot with specified parameters.
    """
    MAIN_AGGREGATOR = 'Make'
    SECONDARY_AGGREGATOR = 'Pool'
    GROUPBY_COLUMNS = [SECONDARY_AGGREGATOR, MAIN_AGGREGATOR]
    COUNT_COLUMN = 'Count'
    REQUIRED_COLUMNS = [MAIN_AGGREGATOR, 'FuelType', 'FuelConsumption', 'Co2EmissionsWltp', 
                        'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']
    DROPNA_SUBSET = ['FuelConsumption']
    AGGREGATION_DICT = {
        MAIN_AGGREGATOR: 'size',
        'FuelType': lambda x: x.mode()[0],
        'FuelConsumption': 'mean',
        'EnginePower': 'mean',
        'Co2EmissionsWltp': 'mean',
        'MassRunningOrder': 'mean',
        'BaseWheel': 'mean',
        'AxleWidthSteering': 'mean',
        'ElectricRange': 'mean'}
    RENAME_COLUMNS = {MAIN_AGGREGATOR: 'Count'}
    DISPLAY_STYLES = generate_styles()
    
    MASS_ENGINE_SCATTER_PARAMS = {
        "x": "MassRunningOrder",
        "y": "EnginePower",
        "color": "Pool",
        "hover_name": "Count",
        "labels": {
            "MassRunningOrder": "Masse du véhicule",
            "EnginePower": "Puissance du moteur",
            },
        "title": "Comparaison de la masse et de la puissance des véhicules (KW)",
        }
    
    FACETPLOT_PARAMS = {
        "x": "EnginePower",
        "y": "Co2EmissionsWltp",
        "size": "FuelConsumption",
        "color": MAIN_AGGREGATOR,
        "facet_col": "Pool",
        "hover_data": [MAIN_AGGREGATOR],
        "dropna_subset": ["FuelConsumption"]
    }

    def __init__(self, df:pd.DataFrame, required_columns=None, dropna=False, dropna_subset=None, 
                 aggregation_dict=None, groupby_columns=None, rename_columns=None):
        """
        Initializes the ManufacturerDataAggregator object.

        Args:
            df (pd.DataFrame): The input DataFrame.
            required_columns (list, optional): List of required columns. Defaults to None.
            aggregation_dict (dict, optional): Dictionary specifying the aggregation functions. Defaults to None.
            groupby_columns (list, optional): List of columns to group by. Defaults to None.
            rename_columns (dict, optional): Dictionary specifying the column renaming. Defaults to None.

        Raises:
            ValueError: If the DataFrame does not contain all the required columns.
        """
        # Checking for eventual custom values
        if required_columns is None:
            required_columns = self.REQUIRED_COLUMNS
        if dropna:
            if dropna_subset is None:
                dropna_subset = self.DROPNA_SUBSET
            df.dropna(subset=dropna_subset, inplace=True)
        if aggregation_dict is None:
            aggregation_dict = self.AGGREGATION_DICT
        if groupby_columns is None:
            groupby_columns = self.GROUPBY_COLUMNS
        if rename_columns is None:
            rename_columns = self.RENAME_COLUMNS

        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        pd.set_option('display.float_format', '{:.3f}'.format)

        self._df = df.groupby(groupby_columns).agg(aggregation_dict)
        self._df = self._df.rename(columns=rename_columns)
        total_count = self._df[self.COUNT_COLUMN].sum()
        market_share = ((self._df[self.COUNT_COLUMN] / total_count) * 100).round(2)
        self._df.insert(1, 'MarketShare(%)', market_share)
        self._df = self._df[self._df[self.COUNT_COLUMN] >= 0.002 * total_count]
                
    def __repr__(self):
        """
        Returns a string representation of the ManufacturerDataAggregator object.

        Returns:
            str: String representation of the ManufacturerDataAggregator object.
        """
        return f"Manufacturers(data={self._df.head()})"
    
    def sort(self, by=None, ascending=False):
        """
        Sorts the data by a specified column.

        Args:
            by (str, optional): The column to sort by. Defaults to self.COUNT_COLUMN.
            ascending (bool, optional): Whether to sort in ascending order. Defaults to False.
        """
        if by is None:
            by = self.COUNT_COLUMN
        self._df.sort_values(by=by, ascending=ascending, inplace=True)

    def display(self, n=None, styles=None, title=None, save=True):
        """
        Displays the data with specified styles and title.

        Args:
            n (int, optional): Number of rows to display. Defaults to None.
            styles (dict, optional): Dictionary specifying the display styles. Defaults to None.
            title (str, optional): Title of the display. Defaults to None.
        """
        if styles is None:
            styles = generate_styles()
        displayer(self._df, n, styles, title, save)
    
    def display_sorted(self, by=None, ascending=False, n=None, styles=None, title=None, save=True):
        """
        Sorts and displays the data.

        Args:
            by (str, optional): The column to sort by. Defaults to self.COUNT_COLUMN.
            ascending (bool, optional): Whether to sort in ascending order. Defaults to False.
            n (int, optional): Number of rows to display. Defaults to None.
            styles (dict, optional): Dictionary specifying the display styles. Defaults to None.
            title (str, optional): Title of the display. Defaults to None.
        """
        if by is None:
            by = self.COUNT_COLUMN
        self.sort(by=by, ascending=ascending)
        self.display(n, styles, title, save=save)
            
    def plot_popular_pool_brands(self, save=True, format='png'):
        """
        Plots the popular fuel types by manufacturer.

        Args:
            save (bool, optional): Whether to save the plot. Defaults to True.
            format (str, optional): Format of the saved plot. Defaults to 'png'.
        """
        # Group by Pool and Make and sum the Counts for each group
        grouped_df = self._df.groupby([self.SECONDARY_AGGREGATOR, self.MAIN_AGGREGATOR])\
            [self.COUNT_COLUMN].sum().reset_index(name='Counts')

        # Create a list of traces
        traces = []
        colors = cl.scales['12']['qual']['Paired']  # Get a list of 12 distinct colors
        make_colors = {make: colors[i % len(colors)] for i, make in enumerate(
            grouped_df[self.MAIN_AGGREGATOR].unique())}  # Create a color map
        
        for make in grouped_df[self.MAIN_AGGREGATOR].unique():
            df = grouped_df[grouped_df[self.MAIN_AGGREGATOR] == make]
            traces.append(go.Bar(x=df[self.SECONDARY_AGGREGATOR], y=df['Counts'],
                                 name=make, marker_color=make_colors[make]))

        layout = go.Layout(barmode='stack', 
                           title="Répartition des immatriculations par marque et groupe automobile",
                           height=700)
        fig = go.Figure(data=traces, layout=layout)
        fig.show()
        save_if_needed(fig, save, format)

            
    def plot_scatter(self, df, plot_params=None, save=True, format='png'):
        """
        Plots a scatter plot with specified parameters.

        Args:
            df (pd.DataFrame): The input DataFrame.
            plot_params (dict, optional): Dictionary specifying the plot parameters. Defaults to None.
        """
        if plot_params is None:
            plot_params = {
                "x": "x_column",
                "y": "y_column",
                "color": "color_column",
                "hover_name": "hover_column",
                "labels": {"x": "X Label", "y": "Y Label"},
                "title": "Default Title"
            }
        fig = px.scatter(
            df, 
            x=plot_params["x"], 
            y=plot_params["y"], 
            color=plot_params["color"], 
            hover_name=plot_params["hover_name"], 
            labels=plot_params["labels"],
            title=plot_params["title"],
        )
        fig = add_legend(fig)
        fig.show()
        save_if_needed(fig, save, format)


    def plot_mass_engine_scatter(self, save=True, format='png'):
        """
        Plots the mass-engine scatter plot.
        """
        unique_df = self._df.reset_index(level=self.SECONDARY_AGGREGATOR)\
            .drop_duplicates(subset=[self.COUNT_COLUMN])
        self.plot_scatter(unique_df, self.MASS_ENGINE_SCATTER_PARAMS, save=save)
        
    def facetplot(self, plot_params=None, save=True, format='png'):
        """
        Plots a facet plot with specified parameters.

        Args:
            plot_params (dict, optional): Dictionary specifying the plot parameters. Defaults to None.
        """
        if plot_params is None:
            plot_params = self.FACETPLOT_PARAMS
        average_df = self._df.reset_index()
        average_df = average_df.dropna(subset=plot_params["dropna_subset"])  # Drop rows with NaN 'FuelConsumption'
        fig = px.scatter(average_df, 
                        x=plot_params["x"], 
                        y=plot_params["y"], 
                        size=plot_params["size"], 
                        color=plot_params["color"], 
                        facet_col=plot_params["facet_col"], 
                        hover_data=plot_params["hover_data"])
        for annotation in fig.layout.annotations:
            annotation.text = annotation.text.split('=')[1]
        fig.show()
        save_if_needed(fig, save, format)
    

class CarDataAggregator:

    ########## Default Class attributes ##########
    MAIN_AGGREGATOR = 'Make'
    SECONDARY_AGGREGATOR = 'CommercialName'
    REQUIRED_COLUMNS = ['Make', 'FuelType', 'FuelConsumption', 'EnginePower', 
                        'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 
                        'AxleWidthSteering', 'ElectricRange']

    GROUPBY_COLUMNS = [MAIN_AGGREGATOR, SECONDARY_AGGREGATOR]
    COUNT_COLUMN = 'CarCount'
    REQUIRED_COLUMNS = [MAIN_AGGREGATOR, 'FuelConsumption', 'Co2EmissionsWltp', 
                        'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']
    DROPNA_SUBSET = ['FuelConsumption']

    AGGREGATION_FUNCTIONS = ['min', q1, 'median', 'mean', q3, 'max']

    AGGREGATION_DICT = {
        SECONDARY_AGGREGATOR: ['size'],
        'Pool': ['first'],
        'FuelType': [lambda x: x.mode()[0]],
        'FuelConsumption': AGGREGATION_FUNCTIONS,
        'EnginePower': AGGREGATION_FUNCTIONS,
        'Co2EmissionsWltp': AGGREGATION_FUNCTIONS,
        'MassRunningOrder': AGGREGATION_FUNCTIONS,
        'BaseWheel': AGGREGATION_FUNCTIONS,
        'AxleWidthSteering': AGGREGATION_FUNCTIONS,
        'ElectricRange': AGGREGATION_FUNCTIONS,
    }

    COLUMN_NAME_MAPPING = {
        'min': '_Min',
        'q1': '_Q1',
        'median': '_Median',
        'mean': '_Mean',
        'q3': '_Q3',
        'max': '_Max',
        'size': '_Size',
        '<lambda>': '_Mode',
        'first': ''
    }

    DISPLAY_STYLES = generate_styles()
    
    SPECS_PARAMS = {
        "index": ['Make', 'CommercialName']
        }

    POLAR_CHARTS_PARAMS = {
        "features": ['MassRunningOrder', 'EnginePower', 'BaseWheel', 'Co2EmissionsWltp', 'ElectricRange'],
        "subplot_titles": 'CommercialName',
        "indexes": ['Make', 'CommercialName'],
        }

    TOP_VEHICLES_PARAMS = {
        "top_count": 3,
        "top_columns": 'CommercialName_Size',
        "middle_columns": 'Co2EmissionsWltp_Max',
        "bottom_columns": 'Co2EmissionsWltp_Min',
        "titles": ["Meilleures ventes 2021", "Voitures les plus polluantes", "Voitures les moins polluantes"]
        }

    FACETPLOT_PARAMS = {
        "x": "EnginePower_Max",
        "y": "Co2EmissionsWltp_Max",
        "size": "CommercialName_Size",
        "color": 'Make',
        "facet_col": 'Pool',
        "facet_col_wrap": 4,
        "hover_data": [MAIN_AGGREGATOR, SECONDARY_AGGREGATOR],
        "dropna_subset": ['FuelConsumption_Max'],
        "title": "Engine Power vs CO2 Emissions"
        }
    
    
   ########## End of Default Class attributes ########## 
   
    def __init__(self, df:pd.DataFrame, required_columns=None, dropna=True, dropna_subset=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.rename_columns = {
            **{
                f"{column}_{operation if not callable(operation) else operation.__name__ if callable(operation) else '<lambda>'}": f"{column}{self.COLUMN_NAME_MAPPING[operation if not callable(operation) else operation.__name__ if callable(operation) else '<lambda>']}"
                for column, operations in self.AGGREGATION_DICT.items()
                for operation in operations
            }
        }

        if required_columns is None:
            required_columns = self.REQUIRED_COLUMNS
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        if dropna:
            if dropna_subset is None:
                dropna_subset = self.DROPNA_SUBSET
            df.dropna(subset=dropna_subset, inplace=True)

        pd.set_option('display.float_format', '{:.3f}'.format)

        self._df = df.groupby(self.GROUPBY_COLUMNS).agg(self.AGGREGATION_DICT)        
        self._df.columns = ['_'.join(col).strip() for col in self._df.columns.values]
        self._df.rename(columns=self.rename_columns, inplace=True)

        self.count_column_name = self.SECONDARY_AGGREGATOR + '_size'
        self.count_column_name = self.rename_columns[self.count_column_name]

        total_count = self._df[self.count_column_name].sum()
        self._df['MarketShare(%)'] = ((self._df[self.count_column_name] / total_count) * 100).round(2)
        # activate the following in case there are too many different cars with low counts
        #self._df = self._df[self._df[self.count_column_name] >= 0.0002 * total_count]     

    def __repr__(self):
        return f"Cars(data={self._df.head()})"
    
    def sort(self, by=None, ascending=False):
        if by is None:
            by = self.COUNT_COLUMN
        self._df.sort_values(by=by, ascending=ascending, inplace=True)
        return self

    def display(self, df=None, n=None, styles=None, title=None, save=True):
        if df is None:
            df = self._df
        df = df.head(n) if n else df
        if styles is None:
            styles = generate_styles()
        displayer(df, styles=styles, title=title, save=save)
        return df

    def display_sorted(self, by=None, ascending=False, n=None, stats=None, styles=None, title=None, save=True):
        if by is None:
            by = self.COUNT_COLUMN
        df_copy = self._df.copy()
        sorted_df = df_copy.sort_values(by=by, ascending=ascending)
        displayed_df = sorted_df.head(n) if n else sorted_df
        if stats is not None:
            stats = ['_' + stat for stat in stats]
            regex_pattern = '|'.join(stats)
            displayed_df = displayed_df.filter(regex=regex_pattern)
        if styles is None:
            styles = generate_styles()
        displayer(displayed_df, n=n, styles=styles, title=title, save=save)
        return displayed_df

    def specs(self, car_names, stats=None, styles=None, params=None, save=True):
        """
        Displays the specifications of selected cars.

        Parameters:
        car_names (list): The names of the cars to display.
        styles (dict, optional): The styles to apply to the displayed data.

        """
        if styles is None:
            styles = generate_styles()
        if params is None:
            params = self.SPECS_PARAMS

        if not isinstance(self._df.index, pd.MultiIndex): # Ensure that self.data is a MultiIndex DataFrame
            self._df.reset_index(inplace=True)
            self._df.set_index(params["index"], inplace=True)

        if params['index'][-1] not in self._df.index.names: # Ensure that 'CommercialName' is one of the index levels
            self._df.reset_index(inplace=True)
            self._df.set_index(params['index'][-1], append=True, inplace=True)

        selected_cars = []
        for car_name in car_names:
            try:
                selected_car = self._df.xs(car_name, level=params['index'][-1]).T.rename(columns=lambda x: car_name)
                selected_cars.append(selected_car)
            except KeyError:
                print(f"Car name '{car_name}' not found in {params['index'][-1]} index.")

        if selected_cars:
            selected_cars_df = pd.concat(selected_cars, axis=1)
            new_rows = []
            for i in range(len(selected_cars_df)):
                current_row = selected_cars_df.iloc[i]
                # Convert the current row to a DataFrame and add it to the new rows
                new_rows.append(current_row.to_frame().T)
                # Add a blank row after the first 2 rows plus the length of AGGREGATION_FUNCTIONS, 
                # and then after every length of AGGREGATION_FUNCTIONS rows
                if (i + 1) == (2 + len(self.AGGREGATION_FUNCTIONS)) or ((i + 1) - (2 + len(self.AGGREGATION_FUNCTIONS))) % len(self.AGGREGATION_FUNCTIONS) == 0:
                    blank_row = pd.DataFrame([['--------'] * len(selected_cars_df.columns)], 
                                index=['--------'], 
                                columns=selected_cars_df.columns)
                    new_rows.append(blank_row)
            selected_cars_df = pd.concat(new_rows)
            if stats is not None:
                stats = ['_' + stat for stat in stats]
                regex_pattern = '|'.join(stats)
                selected_cars_df = selected_cars_df[selected_cars_df.index.str.contains(regex_pattern, na=False)]
            displayer(selected_cars_df, styles=styles, n=None, save=save)
        else:
            print(f"No car names found in {params['index'][-1]} index.")
            return None
            
    def plot_polar_charts(self, selected_vehicles, title='', params=None, save=True, format='png'):
        """
        Plots polar charts for selected vehicles.

        Parameters:
        selected_vehicles (pd.DataFrame): The selected vehicles to plot.
        title (str): The title of the plot.

        """
        if params is None:
            params = self.POLAR_CHARTS_PARAMS

        features = params["features"]

        # Get min and max values from the corresponding _Min and _Max columns
        min_values = self._df[[f + '_Min' for f in features]].min()
        max_values = self._df[[f + '_Max' for f in features]].max()

        subplot_titles = selected_vehicles.index.get_level_values(1).tolist()
        # Calculate the number of rows and columns for the subplot grid
        n_cols = 3
        n_rows = -(-len(subplot_titles) // n_cols)  # Ceiling division
        specs = [[{'type': 'polar'} for _ in range(n_cols)] for _ in range(n_rows)]
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, specs=specs)
        for i, vehicle in enumerate(subplot_titles):   # Add a polar chart for each vehicle
            # Get the min and max values for each feature
            vehicle_df = self._df.xs(vehicle, level=1)
            if vehicle_df.empty:
                print(f"Vehicle '{vehicle}' not found in CommercialName index.")
                continue

            row_data = vehicle_df.iloc[0]
            min_data = [row_data[feature + '_Min'] for feature in features]
            max_data = [row_data[feature + '_Max'] for feature in features]

            # Normalize the data
            normalized_min_data = [(value - min_values[feature + '_Min']) / (max_values[feature + '_Max'] - min_values[feature + '_Min']) * 100 for value, feature in zip(min_data, features)]
            normalized_max_data = [(value - min_values[feature + '_Min']) / (max_values[feature + '_Max'] - min_values[feature + '_Min']) * 100 for value, feature in zip(max_data, features)]

            # Calculate the row and column for this subplot
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Add the min trace
            fig.add_trace(go.Scatterpolar(
                r=normalized_min_data,
                theta=features,
                fill='toself',
                name=vehicle + ' Min',
                showlegend=True  # Hide the automatic legend
            ), row=row, col=col)

            # Add the max trace
            fig.add_trace(go.Scatterpolar(
                r=normalized_max_data,
                theta=features,
                fill='toself',
                name=vehicle + ' Max',
                showlegend=True  # Hide the automatic legend
            ), row=row, col=col)
            fig.update_polars(radialaxis_range=[0, 100])
                        
    
        fig = increase_font_size(fig, font_size=30)        
        fig.update_layout(title_text=title, height=500, width=1200)


        for i in range(1, 4):
            fig.update_polars(radialaxis_range=[0, 100], radialaxis_rotation=20, selector={"polar": f"polar{i}"})
        
        fig = add_legend(fig)
        fig.show()
        save_if_needed(fig, save, format)

    def plot_top_vehicles(self, params=None, save=True, format='png'):
        if params is None:
            params = self.TOP_VEHICLES_PARAMS
        top_3_count = self._df.nlargest(params["top_count"], params["top_columns"])
        top_3_co2 = self._df.nlargest(params["top_count"], params["middle_columns"])
        bottom_3_co2 = self._df.nsmallest(params["top_count"], params["bottom_columns"])

        # Plot polar charts for each selection
        self.plot_polar_charts(top_3_count, params["titles"][0], save=save, format=format)
        self.plot_polar_charts(top_3_co2, params["titles"][1], save=save, format=format)
        self.plot_polar_charts(bottom_3_co2, params["titles"][2], save=save, format=format)
        
    def plot_selected_vehicles(self, vehicle_names, save=False, format='png'):
            selected_vehicles_df = self._df[self._df.index.get_level_values(1).isin(vehicle_names)]
            self.plot_polar_charts(selected_vehicles=selected_vehicles_df, save=save, format=format)
        
    def facetplot(self, params=None, save=True, format='png'):
        """
        Generates a facet plot of engine power vs. CO2 emissions.

        """
        if params is None:
            params = self.FACETPLOT_PARAMS
        average_df = self._df.reset_index()
        average_df = average_df.dropna(subset=params["dropna_subset"])  # Drop rows with NaN 'FuelConsumption'
        fig = px.scatter(average_df, x=params["x"], y=params["y"], size=params['size'], color=params["color"], 
                        facet_col=params["facet_col"], 
                        facet_col_wrap=params["facet_col_wrap"], 
                        hover_data=params["hover_data"])
        for annotation in fig.layout.annotations:
            annotation.text = annotation.text.split('=')[1]
        fig.update_layout(showlegend=False, title_text=params["title"], title_x=0.5, height=1000, width=1000)
        fig.show()
        save_if_needed(fig, save, format)
        
########### End of classes ##########









########## External sources ##########
def get_country_population(): # WIKIPEDIA QUERY
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT ?countryCode ?population
    WHERE
    {
        ?country wdt:P463 wd:Q458; wdt:P297 ?countryCode; wdt:P1082 ?population.
    }
    """

    data = requests.get(url, params={'format': 'json', 'query': query}).json()
    country_population = {item['countryCode']['value']: int(item['population']['value']) for item in data['results']['bindings']}
    return country_population #dict

def get_country_gdp(): # WIKIPEDIA QUERY
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT ?countryCode ?gdp
    WHERE
    {
        ?country wdt:P463 wd:Q458; wdt:P297 ?countryCode; wdt:P2131 ?gdp.
    }
    """

    data = requests.get(url, params={'format': 'json', 'query': query}).json()
    country_gdp = {item['countryCode']['value']: float(item['gdp']['value']) for item in data['results']['bindings']}
    return country_gdp #dict

def get_country_stats(): # WIKIPEDIA QUERY
    country_population = get_country_population()
    country_gdp = get_country_gdp()

    country_stats = {}
    for countryCode in country_population.keys():
        if countryCode in country_gdp:
            country_stats[countryCode] = {
                'population': country_population[countryCode],
                'gdp_per_capita': country_gdp[countryCode] / country_population[countryCode]
            }

    return country_stats

# Erreur de virgule sur les données de la Hongrie, les stats seront passées en variable:


########## End External sources ##########