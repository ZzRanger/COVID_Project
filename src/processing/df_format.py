""" This module formats processed data for use in the SIR model"""
from pandas import read_csv
from pandas import DataFrame

def sir_format(s_path, i_path, r_path):
    """ This function takes in processed data for the SIR model, 
        reformats it, then returns it

        Arguments
        ---------
        s_path (str): path for csv file w/ % of pop. suceptible for countries
        i_path (str): path for csv file w/ % of pop. infected for countries
        r_path (str): path for csv file w/ % of pop. recovered for countries
        
        Returns
        -------
        s_countries (pd.DataFrame): DataFrame w/ % of pop. suceptible for countries
        i_countries (pd.DataFrame): DataFrame w/ % of pop. infected for countries
        r_countries (pd.DataFrame): DataFrame w/ % of pop. recovered for countries
    """
    s_countries = read_csv(s_path)
    s_countries.index = s_countries['Unnamed: 0'].values
    s_countries = s_countries.drop(axis = 1, labels = 'Unnamed: 0')

    i_countries = read_csv(i_path)
    i_countries.index = i_countries['Unnamed: 0'].values
    i_countries = i_countries.drop(axis = 1, labels = 'Unnamed: 0')

    r_countries = read_csv(r_path)
    r_countries.index = r_countries['Unnamed: 0'].values
    r_countries = r_countries.drop(axis = 1, labels = 'Unnamed: 0')
    
    return s_countries, i_countries, r_countries

def jhu_format(confirmed, deaths, recovered):
    """ This function takes in processed JHU ,
        reformats it, then returns it

        Arguments
        ---------
        confirmed (str): path for csv file w/ confirmed cases for countries
        deaths (str): path for csv file w/ # of deaths for countries
        recovered (str): path for csv file w/ recovered cases countries
        
        Returns
        -------
        covid_confirmed (pd.DataFrame): DataFrame w/ confirmed cases for countries
        covid_deaths (pd.DataFrame): DataFrame w/ # of deaths for countries
        covid_recovered (pd.DataFrame): DataFrame w/ recovered cases countries
    """
    covid_confirmed = read_csv(confirmed)
    covid_confirmed.index = covid_confirmed['Country/Region'].values
    covid_confirmed = covid_confirmed.drop(axis = 1, labels = 'Unnamed: 0')

    covid_deaths = read_csv(deaths)
    covid_deaths.index = covid_deaths['Country/Region'].values
    covid_deaths = covid_deaths.drop(axis = 1, labels = 'Unnamed: 0')

    covid_recovered = read_csv(recovered)
    covid_recovered.index = covid_recovered['Country/Region'].values
    covid_recovered = covid_recovered.drop(axis = 1, labels = 'Unnamed: 0')

    return covid_confirmed, covid_deaths, covid_recovered