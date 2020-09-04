""" This module formats processed data for 
    use in US State COVID Visualizations """
# Basic Library Imports
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame

state_abbrev_only = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL',
'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}

us_state_abbrev = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 
'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',    'Iowa': 'IA', 'Kansas': 'KS',
'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
'Puerto Rico': 'PR', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}

def import_jhu_data(death_path, confirmed_path):
    """ This function imports raw US JHU data, 
        then cleans and returns the data

        Arguments
        ---------
        death_path (str): path for csv file w/ # of deaths
        confirmed_path (str): path for csv file w/ confirmed cases

        Returns
        -------
        states_confirmed (pd.DataFrame): DataFrame w/ confirmed cases
        states_deaths (pd.DataFrame): DataFrame w/ # of deaths
        state_population (dict): index is state name and value is state population
    """
    raw_us_deaths = read_csv(death_path)
    raw_us_confirmed = read_csv(confirmed_path)

    # Set up states dataframe
    index = np.unique(raw_us_confirmed['Province_State'].values)
    indices = [11 + i for i in range(len(raw_us_confirmed.columns) - 11)]
    columns = list(raw_us_confirmed.columns[indices])
    states_confirmed, states_deaths = DataFrame(index = index, columns = columns), DataFrame(index = index, columns = columns)

    # Fill in values
    for i in states_confirmed.columns:
        for j in states_confirmed.index:
            # Confirmed data
            state = raw_us_confirmed[raw_us_confirmed['Province_State'] == j]
            states_confirmed.loc[j, i] = state.loc[:, i].sum()
            # Deaths data
            state = raw_us_deaths[raw_us_deaths['Province_State'] == j]
            states_deaths.loc[j, i] = state.loc[:, i].sum()

    # US State Population
    state_population = dict({x: None for x in states_confirmed.index})
    for i in state_population.keys():
        state = raw_us_deaths[raw_us_deaths['Province_State'] == i]
        state_population[i] = state.loc[:, 'Population'].sum()

    return states_confirmed, states_deaths, state_population

def import_hospital_data(us_path, state_path, columns):
    """ This function imports raw US hospital data, 
        then cleans and returns the data

        Arguments
        ---------
        us_path (str): path for csv file w/ US hospital info
        state_path (str): path for csv file w/ state hospital info
        columns (pandas.core.indexes.base.Index): Index of dates starting from 1 / 22

        Returns
        -------
        us_hospital_data (pd.DataFrame): DataFrame w/ US hospital data
        hospital_data (pd.DataFrame): DataFrame w/ state hospital data
    """
    raw_us_hospital_data = read_csv(us_path)
    raw_state_hospital_data = read_csv(state_path)

    us_hospital_data = DataFrame(index = raw_us_hospital_data.columns, columns = columns)
    for i in raw_us_hospital_data.index:
        x = raw_us_hospital_data.loc[i, :]
        y = str(int(x.date))
        if '0' == y[-2]:
            date = y[-3] + '/' + y[-1] + '/' + y[:2]
            
        else:
            date = y[-3] + '/' + y[-2:] + '/' + y[:2]

        for j in x.index:
            us_hospital_data.loc[j, date] = x[j]

    # Create dictionary for each hospital data column & corresponding dataframe
    hospital_data = dict({x: None for x in raw_state_hospital_data.columns[2:]})
    for i in raw_state_hospital_data.columns[2:]:
        # Create DataFrame for each column
        df = DataFrame(index = us_state_abbrev.keys(), columns = columns)
        
        for state in us_state_abbrev.keys():

            abbr = us_state_abbrev[state]
        # Create dataframe w/ all entries from state
            x = raw_state_hospital_data[raw_state_hospital_data['state'] == abbr]

            # Copy all values from original dataframe to state dataframe
            for j in x.index:
                y = str(x['date'][j])
                if '0' == y[-2]:
                    date = y[-3] + '/' + y[-1] + '/' + y[:2]
                    
                else:
                    date = y[-3] + '/' + y[-2:] + '/' + y[:2]
            
                df.loc[state, date] = x.loc[j, i]

        # Assign dataframe as value for state in dictionary
        hospital_data[i] = df

    return us_hospital_data, hospital_data