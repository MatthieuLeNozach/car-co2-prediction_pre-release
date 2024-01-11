import pandas as pd
import requests
import re
import io

########## Classes ##########
class Countries:
    """_summary_
    """
    def __init__(self, df:pd.DataFrame, country_stats: dict):
        required_columns = ['Country', 'Make', 'FuelType', 'FuelConsumption', 
                            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")
        
        
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
        self.data = self.data.reset_index()
        
        self.data['Population'] = self.data['Country'].map(country_population)
        pop = self.data.pop('Population')
        self.data.insert(1, 'Population', pop)
        
        self.data['GdpPerCapita'] = self.data['Country'].map(country_gdp_per_capita)
        pop = self.data.pop('GdpPerCapita')
        self.data.insert(2, 'GdpPerCapita', pop)



class Manufacturers:
    def __init__(self, df:pd.DataFrame):
        required_columns = [
            'Make', 'FuelType', 'FuelConsumption', 
            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']

        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        self.data = df.groupby(['Pool', 'Make']).agg({
            'Make': 'count',
            'Pool': 'count',
            'FuelType': lambda x: x.mode()[0],
            'FuelConsumption': 'mean',
            'EnginePower': 'mean',
            'Co2EmissionsWltp': 'mean',
            'MassRunningOrder': 'mean',
            'BaseWheel': 'mean',
            'AxleWidthSteering': 'mean'})
    

        

########### End of classes ##########



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