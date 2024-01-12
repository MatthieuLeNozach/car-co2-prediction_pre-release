import numpy as np
import pandas as pd
import requests
import io
from auto_co2.styles import generate_styles, displayer


########## Classes ##########
class Countries:
    """_summary_
    """
    def __init__(self, df:pd.DataFrame, country_stats: dict):
        required_columns = ['Country', 'Make', 'FuelType', 'FuelConsumption', 
                            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")
        
        pd.set_option('display.float_format', '{:.3f}'.format)       
        
        self.data = df.groupby('Country').agg({
            'ID': 'count',
            'Make': lambda x: x.mode()[0],
            'FuelType': lambda x: x.mode()[0],
            'FuelConsumption': 'mean',
            'EnginePower': 'mean',
            'Co2EmissionsWltp': 'mean',
            'MassRunningOrder': 'mean',
            'BaseWheel': 'mean',
            'AxleWidthSteering': 'mean'})
        
        
        self.data = self.data.rename(columns={'ID': 'Count'})
        
        country_stats = pd.DataFrame(country_stats).transpose()
        country_stats.index.name = 'Country'
        self.data = self.data.join(country_stats, how='inner')

        gdp = self.data.pop('gdp_per_capita')
        self.data.insert(0, 'gdp_per_capita', gdp)
        
        pop = self.data.pop('population')        
        self.data.insert(0, 'population', pop)

        # Convert the index to a column
        self.data.reset_index(inplace=True)
                
    def __repr__(self):
        return f"Countries(data={self.data.head()})"

    def display(self, n=None, styles=None):
        if styles is None:
            styles = generate_styles()
        displayer(self.data, n, styles)



class Manufacturers:
    def __init__(self, df:pd.DataFrame):
        required_columns = [
            'Make', 'FuelType', 'FuelConsumption', 
            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']

        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        pd.set_option('display.float_format', '{:.3f}'.format)

        self.data = df.groupby(['Pool', 'Make']).agg({
            'Make': 'count',
            'FuelType': lambda x: x.mode()[0],
            'FuelConsumption': 'mean',
            'EnginePower': 'mean',
            'Co2EmissionsWltp': 'mean',
            'MassRunningOrder': 'mean',
            'BaseWheel': 'mean',
            'AxleWidthSteering': 'mean'})
        
        self.data = self.data.rename(columns={'Make': 'Count'})
        
        total_count = self.data['Count'].sum()
        market_share = ((self.data['Count'] / total_count) * 100).round(2)
        self.data.insert(1, 'MarketShare(%)', market_share)
                
    def __repr__(self):
        return f"Manufacturers(data={self.data.head()})"

    def display(self, n=None, styles=None):
        if styles is None:
            styles = generate_styles()
        displayer(self.data, n, styles)

        

########### End of classes ##########




########## Toolkit ##########
def float_rounder(df, n=3):
        float_cols = df.select_dtypes(include='float64').columns
        return df[float_cols].round(n)



########## End of toolkit ##########





########## Aggregation tools ##########
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

########## End aggregation tools ##########