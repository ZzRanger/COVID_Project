""" This module runs the code for the Mask Mandate Section of US State COVID Trends Visualizations.ipynb """
# Basic Library Imports
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame

def mask_mandate_dates(us_state_abbrev):
    """ This function creates a dictionary for when 
        states passed a statewide mask mandate

        Arguments
        ---------
        us_state_abbrev (dict): dict of US state names and abbrev
            {name}: {abbrev}
        
        Returns
        -------
        state_mask_mandates (dict): dict of US state names and mandate date
            {state name}: {state mask mandate date}
    """
    # Dictionary of Mask Mandate Dates in US
    state_names = list(us_state_abbrev.keys())
    state_names.sort()

    state_mask_mandates = dict({x: None for x in state_names})

    dates = ['7/16/20', None, None, '7/20/20', '6/18/20', '7/17/20', 
            '4/20/20', '4/28/20', '5/16/20', None, None, '4/20/20', 
            None, '5/1/20', '7/27/20', None, '7/3/20', '5/11/20', 
            '7/13/20', '5/1/20', '4/18/20', '5/6/20', '6/18/20', 
            '7/25/20', '8/4/20', None, '7/16/20', None, '6/24/20', 
            '8/11/20', '4/8/20', '5/16/20', '4/17/20', '6/26/20', 
            None, '7/23/20', None, '7/1/20', '4/19/20', '4/30/20',
            '5/8/20', None, None, None, '7/3/20', None, '8/1/20',
            '5/29/20', '6/26/20', '7/6/20', '8/1/20', None]

    counter = 0
    for state in state_mask_mandates.keys():
        state_mask_mandates[state] = dates[counter]
        counter += 1

    return state_mask_mandates

def graph_ax(ax, hospital_data, label, states, state_population):
    """ This function takes in an AxesSubplot object, adds individual 
        state graphs for daily cases & pos. rate and graph properties, 
        then returns it

        Arguments
        ---------
        ax (matplotlib.axes._subplots): blank AxesSubplot object
        hospital_data (dict): dictionary containing DataFrames w/ hospital data
        label (str): type of graph
        states (list): list of states
        state_population (dict): dictionary w/ state names & its population
            {state name}: {state population}
        
        Returns
        -------
        ax (matplotlib.axes._subplots): AxesSubplot object
    """

    dates = hospital_data['positive'].columns[100:]

    tick_loc = [i * 7 for i in range(len(dates) // 7 + 1)]
    labels = [dates[7 * i] for i in range(len(dates) // 7 + 1)]

    # Set xticks
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(labels = labels, rotation = 60)

    for state in states:
        # Create data
        if label == 'positive':
            data = (hospital_data['positiveIncrease'].loc[state, '5/1/20':].rolling(window = 7).sum() / 7) * 100 / state_population[state]
        elif label == 'totalIncr':
            data = (hospital_data['totalTestResultsIncrease'].loc[state, '5/1/20':].rolling(window = 7).sum() / 7) * 100 / state_population[state]
        elif label == 'percent_pos':
            data = (hospital_data['positiveIncrease'].loc[state, '5/1/20':].rolling(window = 7).sum() / 7) * 100 / (hospital_data['totalTestResultsIncrease'].loc[state, '5/1/20':].rolling(window = 7).sum() / 7)
        else:
            return "Invalid label"

        t = np.linspace(0, len(data), len(data))
        ax.plot(t, data, label = state)

    ax.legend()
    return ax

def states_graph_ax(ax, hospital_data, label, state_masks, state_population):
    """ This function takes in an AxesSubplot object, adds average state 
        graphs for daily cases & pos. rate for each category and graph 
        properties, then returns it

        Arguments
        ---------
        ax (matplotlib.axes._subplots): blank AxesSubplot object
        hospital_data (dict): dictionary containing DataFrames w/ hospital data
        label (str): type of graph
        state_masks (list): list of states
        state_population (dict): dictionary w/ state names & its population
            {state name}: {state population}
        
        Returns
        -------
        ax (matplotlib.axes._subplots): AxesSubplot object
    """

    names = ['Before May 1st', 'May', 'June', 'July', 'August', 'No Mask Mandate']

    # Set dates
    date_index = 131
    t = np.linspace(0, len(hospital_data['positive'].columns[date_index:]), len(hospital_data['positive'].columns[date_index:]))
    date_name = '6/1/20'

    # Tick labels
    tick_loc = [7 * i for i in range(len(t) // 7)]
    tick_loc.append(len(t) - 1)
    dates = list(hospital_data['positive'].columns[date_index:])
    tick_label = [dates[i] for i in tick_loc]

    total_data = []

    for j in range(len(state_masks)):
        
        states = state_masks[j]
        for state in states:
            data = mask_data(label, hospital_data, state_population, state, date_name)

            if total_data == []:
                total_data.extend(data)
            else:
                total_data = [total_data[i] + data[i] for i in range(len(data))]


        total_data = [total_data[i] / len(states) for i in range(len(total_data))]

        ax.plot(t, total_data, label = names[j])

    ax.set_xticks(tick_loc)
    ax.set_xticklabels(tick_label, rotation = 50)
    ax.legend()

    return ax

def mask_data(label, hospital_data, state_population, state, date_name):
    """ This function calculates the 7 day rolling average data for a state

        Arguments
        ---------
        label (str): type of graph
        hospital_data (dict): dictionary containing DataFrames w/ hospital data
        state_population (dict): dictionary w/ state names & its population
            {state name}: {state population}        
        state (str): state name
        date_name (str): start date
        
        Returns
        -------
        data (list): 7 day rolling average data
    """
    if label == 'pos_rate':
        # Process Data
        pos = (hospital_data['positiveIncrease'].loc[state, date_name:].rolling(window = 7).sum() / 7).ffill()
        total = (hospital_data['totalTestResultsIncrease'].loc[state, date_name:].rolling(window = 7).sum() / 7).ffill()
        data = list(pos * 100 / total)
    
    elif label == 'cases':
        # Process data
        data = list((hospital_data['positiveIncrease'].loc[state, date_name:].rolling(window = 7).sum() / 7).ffill())
        population = state_population[state]
        data = [data[i] * 100 / population for i in range(len(data))]

    return data
