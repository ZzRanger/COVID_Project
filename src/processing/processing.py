""" This module runs the data processing code for Processing.ipynb """
# Import libraries
from pandas import read_csv
import pandas as pd
from pandas import DataFrame as df

def import_data(confirmed, deaths, recovered): 
        """ This function imports raw global JHU data,
            then cleans and returns the data

            Arguments
            ---------
            confirmed (str): path for csv file w/ confirmed cases
            deaths (str): path for csv file w/ # of deaths
            recovered (str): path for csv file w/ recovered cases

            Returns
            -------
            raw_covid_confirmed(pd.DataFrame): DataFrame w/ confirmed cases
            raw_covid_deaths(pd.DataFrame): DataFrame w/ # of deaths
            raw_covid_recovered(pd.DataFrame): DataFrame w/ recovered cases
        """
        # COVID Data
        raw_covid_confirmed = read_csv(confirmed)
        raw_covid_deaths = read_csv(deaths)
        raw_covid_recovered = read_csv(recovered)

        # Remove Cruise Ships
        remove_index = []
        remove_index.append(raw_covid_confirmed[raw_covid_confirmed['Country/Region'] == 'Diamond Princess'].index[0])
        remove_index.append(raw_covid_confirmed[raw_covid_confirmed['Country/Region'] == 'MS Zaandam'].index[0])
        raw_covid_confirmed = raw_covid_confirmed.drop(index = remove_index)
        raw_covid_deaths = raw_covid_deaths.drop(index = remove_index)
        raw_covid_recovered = raw_covid_recovered.drop(index = remove_index)

        return raw_covid_confirmed, raw_covid_deaths, raw_covid_recovered

def process_sir(confirmed, deaths, recovered, country_pop):
    """ This function takes in clean JHU data from import_data 
        and returns data processed for the SIR model

        Arguments
        ---------
        confirmed (pd.DataFrame): DataFrame w/ confirmed cases
        deaths (pd.DataFrame): DataFrame w/ # of deaths
        recovered (pd.DataFrame): DataFrame w/ recovered cases

        Returns
        -------
        s_countries(pd.DataFrame): DataFrame w/ % of pop. suceptible for countries
        i_countries(pd.DataFrame): DataFrame w/ % of pop. infected for countries
        r_countries(pd.DataFrame): DataFrame w/ % of pop. recovered for countries
    """
        
    countries_name = list(set(confirmed['Country/Region'].sort_values()))
    countries_name.sort()

    # SIR Dataframe
    s_countries = df(index = countries_name.sort(), columns = confirmed.columns[4:])
    i_countries = df(index = countries_name.sort(), columns = confirmed.columns[4:])
    r_countries = df(index = countries_name.sort(), columns = confirmed.columns[4:])

    try:
        for country in countries_name:
            # Check for value error
            if country not in country_pop:
                continue
            # Iterate for each date
            for i in confirmed.columns[4:]:
                # Suceptible (pop - confirmed - recovered - deaths)
                s_countries.loc[country, i] = (country_pop[country] * 1e6 - int(confirmed[confirmed['Country/Region'] == country].loc[:, i].sum()) - int(recovered[recovered['Country/Region'] == country].loc[:, i].sum()) - int(deaths[deaths['Country/Region'] == country].loc[:, i].sum())) / (country_pop[country] * 1e6)
                # Infected (confirmed - recovered - deaths)
                i_countries.loc[country, i] = (int(confirmed[confirmed['Country/Region'] == country].loc[:, i].sum()) - int(recovered[recovered['Country/Region'] == country].loc[:, i].sum()) - int(deaths[deaths['Country/Region'] == country].loc[:, i].sum())) / (country_pop[country] * 1e6)
                # Recovered (recovered + deaths)
                r_countries.loc[country, i] = (int(recovered[recovered['Country/Region'] == country].loc[:, i].sum()) + int(deaths[deaths['Country/Region'] == country].loc[:, i].sum())) / (country_pop[country] * 1e6)
    except KeyError:
        pass
    return s_countries, i_countries, r_countries

def pop(raw_country_pop):
    """ This function takes in raw country population data and 
        returns a dictionary w/ country populations

        Arguments
        ---------
        confirmed (str): path for csv file w/ confirmed cases
        deaths (str): path for csv file w/ # of deaths
        recovered (str): path for csv file w/ recovered cases

        Returns
        -------
        country_pop (dict): index is country name and value is 
                            population in millions
    """
    # Import World Population as Dictionary
    import_country_pop = read_csv(raw_country_pop)
    country_pop = dict()

    # Population in millions
    for i in range(len(import_country_pop.index)):
        country_pop[import_country_pop.iloc[i, 0]] = float(import_country_pop.iloc[i,1]) / 1000

    # West Bank and Gaza
    country_pop['West Bank and Gaza'] = 4.569  

    # Kosovo
    country_pop['Kosovo'] = 1.845

    return country_pop