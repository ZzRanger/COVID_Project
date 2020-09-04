""" This module runs the code for the General Trends Section of US State COVID Trends Visualizations.ipynb """
# Basic Library Imports
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame
import plotly.express as px  # Be sure to import express

def gen_us_trends(us_hospital_data):
    """ This function plots US Daily Cases & % Pos. Rate of COVID Tests

    Arguments
    ---------
    us_hospital_data (pd.DataFrame): DataFrame w/ US hopsital data

    Returns
    -------
    fig (matplotlib.pyplot.figure): graph figure
    ax (matplotlib.axes._subplots): AxesSubplot object
    """
    # US Trends
    fig = plt.figure(figsize = (15, 24))
    dates = us_hospital_data.columns[100:]

    ax = fig.add_subplot(211)
    data = us_hospital_data.loc['positiveIncrease', '5/1/20':].rolling(window = 7).sum() / 7
    t = np.linspace(0, len(data), len(data))
    ax.plot(t, data)

    tick_loc = [i * 7 for i in range(len(data) // 7)]
    tick_loc.append(len(data))
    labels = [dates[7 * i] for i in range(len(data) // 7)]
    labels.append(dates[-1])

    # Set xticks
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(labels = labels, rotation = 90)

    # Set title
    ax.set_title('US Daily COVID Cases')
    ax.set_ylabel('# of Cases')
    ax.set_xlabel('Date')

    # US Pos. Rate
    ax = fig.add_subplot(212)
    data = (us_hospital_data.loc['positiveIncrease', '5/1/20':].rolling(window = 7).sum() / 7) / (us_hospital_data.loc['totalTestResultsIncrease', '5/1/20':].rolling(window = 7).sum() / 7)
    t = np.linspace(0, len(data), len(data))
    ax.plot(t, data)

    # Set xticks
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(labels = labels, rotation = 90)

    # Set title
    ax.set_title('US % Positive Rate')
    ax.set_ylabel('% Tests Positive')
    ax.set_xlabel('Date')

    return fig, ax

def gen_state_trends(hospital_data, state_abbrev_only):
    """ This function finds weekly state trends for daily cases and
        % Pos. Rate of COVID Tests

        Arguments
        ---------
        hospital_data (dict): dictionary containing DataFrames w/ hospital data
        state_abbrev_only (dict): dict of US state names and abbrev
            {name}: {abbrev}

        Returns
        -------
        state_df (pd.DataFrame): DataFrame w/ state daily case trends
        pos_df (pd.DataFrame): DataFrame w/ state % Pos. Rate trends
    """
    # General
    dates = [-126, -120, -119, -113, -112, -106, -105, -99, -98, 
            -92, -91, -85, -84, -78, -77, -71, -70, -64, -63, -57, -56, -50, 
            -49, -43, -42, -36, -35, -29, -28, -22, -21, -15, -14, -8, -7, -1]
    dates = [hospital_data['positive'].columns[x] for x in dates]

    states_list = [x for x in state_abbrev_only.keys()]

    # Positive Cases 
    state_df = DataFrame(columns = ["state", "value", "date", "hover_name"])
    # state_df.state = [x for x in state_abbrev_only.values()]

    counter = 0
    for date_index in range(len(dates[2:]) // 2):
        for state in state_abbrev_only.keys():
            # JHU
            # current_wk = (states_confirmed.loc[state, '8/22/20'] - states_confirmed.loc[state,'8/15/20']) 
            # previous_wk = (states_confirmed.loc[state, '8/14/20'] - states_confirmed.loc[state, '8/7/20']) 

            # COVID Tracker Data
            current_wk = hospital_data['positiveIncrease'].loc[state, dates[date_index * 2 + 2]:dates[date_index * 2 + 3]].sum() / 7
            previous_wk = hospital_data['positiveIncrease'].loc[state, dates[date_index * 2]:dates[date_index * 2 + 1]].sum() / 7
            # print(state + ' PosIncr', current_wk, previous_wk)
            difference = (current_wk - previous_wk) / previous_wk * 100
            state_df.loc[counter, 'date'] = dates[date_index * 2 + 3]
            state_df.loc[counter, 'value'] = difference
            state_df.loc[counter, 'state'] = state_abbrev_only[state]
            state_df.loc[counter, 'hover_name'] = state
            counter += 1
    
    state_df.sort_index(ascending = True)
    # %Pos.
    pos_df = DataFrame(columns = ["state", "value", "date", "hover_name"])
    counter = 0
    for date_index in range(len(dates[2:]) // 2):
        for state in state_abbrev_only.keys():
            # JHU
            # current_wk = (states_confirmed.loc[state, '8/22/20'] - states_confirmed.loc[state,'8/15/20']) 
            # previous_wk = (states_confirmed.loc[state, '8/14/20'] - states_confirmed.loc[state, '8/7/20']) 

            # COVID Tracker Data
            current_wk = (hospital_data['positiveIncrease'].loc[state, dates[date_index * 2 + 2]:dates[date_index * 2 + 3]].sum() / 7) / (hospital_data['totalTestResultsIncrease'].loc[state, dates[date_index * 2 + 2]:dates[date_index * 2 + 3]].sum() / 7)
            previous_wk = (hospital_data['positiveIncrease'].loc[state, dates[date_index * 2]:dates[date_index * 2 + 1]].sum() / 7) / (hospital_data['totalTestResultsIncrease'].loc[state, dates[date_index * 2]:dates[date_index * 2 + 1]].sum() / 7)
            # print(state + ' PosIncr', current_wk, previous_wk)
            difference = (current_wk - previous_wk) / previous_wk * 100
            pos_df.loc[counter, 'date'] = dates[date_index * 2 + 3]
            pos_df.loc[counter, 'value'] = difference
            pos_df.loc[counter, 'state'] = state_abbrev_only[state]
            pos_df.loc[counter, 'hover_name'] = state
            counter += 1

    """
    # Create Dictionaries of State Trends
    dates = [(1 + 7 * i) * -1 for i in range(10)]
    dates = [hospital_data['positive'].columns[x] for x in dates]

    reverse_dates = dates

    # Daily Cases
    case_trends = dict({x + -10: [] for x in range(20)})

    def find_trend(state, counter, check): 
        df = state_df[state_df['hover_name'] == state]
        val = df[df['date'] == reverse_dates[counter]].value.values[0]
        if val > 0:
            return counter if check == -1 else find_trend(state, counter + 1, 1) 

        else:
            return counter * -1 if check == 1 else find_trend(state, counter + 1, -1)            
            
    for state in states_list:
        case_trends[find_trend(state, 0, 0)].append(state)

    # Pos. Rate
    pos_rate_trends = dict({x + -10: [] for x in range(20)})

    def find_pos_trend(state, counter, check): 
        df = pos_df[pos_df['hover_name'] == state]
        val = df[df['date'] == reverse_dates[counter]].value.values[0]
        if val > 0:
            return counter if check == -1 else find_pos_trend(state, counter + 1, 1) 

        else:
            return counter * -1 if check == 1 else find_pos_trend(state, counter + 1, -1)            
            
    for state in states_list:
        pos_rate_trends[find_pos_trend(state, 0, 0)].append(state)

    for num in list(case_trends.keys()):
        if case_trends[num] == []: del case_trends[num] 
        if pos_rate_trends[num] == []: del pos_rate_trends[num] 
"""
    return state_df, pos_df # , case_trends, pos_rate_trends
            
def state_heat_map(state_df, pos_df):
    """ This function graphs daily case and % Pos. Rate trends on an animated Heat Map

    Arguments
    ---------
    state_df (pd.DataFrame): DataFrame w/ state daily case trends
    pos_df (pd.DataFrame): DataFrame w/ state % Pos. Rate trends   

    Returns
    -------
    fig, fig2 (matplotlib.pyplot.figure): graph figures for daily case & pos. rate
    """
    state_df['value'] = state_df['value'].astype(float)
    pos_df['value'] = pos_df['value'].astype(float)
    # State Heat Map
    
    # Find max & min values 
    max = 50
    min = -50

    fig = px.choropleth(state_df,  # Input Pandas DataFrame
                        locations="state",  # DataFrame column with locations
                        color="value",  # DataFrame column with color values
                        range_color = [min, max],
                        color_continuous_scale= 'bluered',
                        animation_frame = 'date',
                        labels = {'value': '% Change'},
                        color_continuous_midpoint = 0,
                        hover_name = "hover_name", # DataFrame column hover info
                        locationmode = 'USA-states') # Set to plot as US States
    fig.update_layout(
        title_text = 'Weekly % Change of COVID Cases in US States', # Create a Title
        geo_scope='usa',  # Plot only the USA instead of globe
    )

    # Controls animation speed
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

    max = 50
    min = -50

    fig2 = px.choropleth(pos_df,  # Input Pandas DataFrame
                        locations="state",  # DataFrame column with locations
                        color="value",  # DataFrame column with color values
                        range_color = [min, max],
                        color_continuous_scale= 'bluered' ,
                        animation_frame = 'date',
                        labels = {'value': '% Change'},
                        color_continuous_midpoint = 0,
                        hover_name = "hover_name", # DataFrame column hover info
                        locationmode = 'USA-states') # Set to plot as US States
    fig2.update_layout(
        title_text = 'Weekly % Change of COVID Pos. Rate in US States', # Create a Title
        geo_scope='usa',  # Plot only the USA instead of globe
    )

    # Controls animation speed
    fig2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

    return fig, fig2