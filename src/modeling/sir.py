""" This module runs the model and graph code for SIR.ipynb """

import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
import scipy.optimize as optimize
import numpy as np
import pandas as pd
from math import sqrt
from datetime import timedelta, date


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def make_dates():
    """ This function creates a list of dates between
        6/1/20 and 6/30/20 

        Returns
        -------
        dates (list): list of dates
    """
    dates = []
    start_dt = date(2020, 6, 1)
    end_dt = date(2020, 6, 30)
    for dt in daterange(start_dt, end_dt):
        dates.append(dt.strftime("%m/%d/%Y"))
    
    for day in range(len(dates)):
        dates[day] = dates[day][1:]
        if dates[day][2] == '0':
            dates[day] = dates[day][0:2] + dates[day][3:]
        dates[day] = dates[day][:-2]
    return dates

def rmse(inpt):
    """ This function is the loss function used by scipy.optimize.minimize

        Arguments
        ---------
        inpt (list): list with the following structure [b, g]
            b and g are variables in the SIR model
            
            For more information about the model, see 
            Background Information in SIR.ipynb 
        
        Returns
        -------
        float: rmse of SIR model
    """
    # int('test')
    b = inpt[0]
    g = inpt[1]
    # int(b, g)
    S, I, R = model(b, g)

    sse = 0
    
    for i in range(len(t) - 1):
        error_one = abs(I[i] - country_infected[i])**2
        error_two = abs(R[i] - country_recovered[i])**2
        sse += (error_one + error_two)
    return sqrt(sse / len(t))

# Total population, N.
N = 1
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3.652613172721974e-06, 0 
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate (beta), and mean recovery rate (gamma; in 1/days)
    
def model(b, g):
    """ This function inputs the following variables into the 
        SIR model and returns the result.
        
        For more information about the model, see 
        Background Information in SIR.ipynb 

        This is used in loss function rmse(inpt) to calculate loss
        
        Arguments
        ---------
        b, g (float): variables in equation 

        Returns
        -------
        S, I, R (np.ndarray): array of y values for S, I, R
    """
    beta, gamma = b, g 
    
    # The SIR model differential equations.
    def deriv(y, time, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, time, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R

def scipy_model(b, g, time):
    """ This function inputs the following variables into the 
        SIR model and returns the result.
        
        For more information about the model, see 
        Background Information in SIR.ipynb 

        Arguments
        ---------
        b, g (float): variables in equation 
        time (np.ndarray): array of x values 

        Returns
        -------
        S, I, R (np.ndarray): array of y values for S, I, R
    """
    beta, gamma = b, g 
    
    # The SIR model differential equations.
    def deriv(y, time, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, time, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R

"""
def sample_model(b, g):
    beta, gamma = b, g 
    t = np.linspace(0, 50, 50)
    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    print(type(y0))
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R
"""

def graph_model(b, g, time):
    """ This function graphs the SIR model

        Arguments
        ---------
        b, g (float): variables for SIR model
        time (np.ndarray): array of x values

        Returns
        -------
        fig (matplotlib.pyplot.figure): graph figure
    """
    S, I, R = scipy_model(b, g, time)
    fig = plt.figure(figsize = (12, 8))
    t = np.linspace(0, len(S), len(S))
    ax = fig.add_subplot()
    ax.plot(t, S, label = "Suceptible")
    ax.plot(t, I, label = "Infected")
    ax.plot(t, R, label = "Recovered")
    ax.set_title("Example SIR Graph")
    ax.set_ylabel("% of population")
    ax.set_xlabel("Dates")
    ax.legend()
    return fig, ax

def sir_scipy(country, i_countries, r_countries, cutoff, x0):
    """ This function fits SIR country data to the SIR model.
        
        For more information about the equation, see 
        Background Information in SIR.ipynb 

        Arguments
        ---------
        country (str): name of country
        i_countries (pd.DataFrame): DataFrame w/ % of pop. infected for countries
        r_countries (pd.DataFrame): DataFrame w/ % of pop. recovered for countries

        Returns
        -------
        val.x (np.ndarray): array w/ optimal values of b & g 
    """

    # Set global variables
    global t, time, country_infected, country_recovered

    # Get rid of memory warning
    plt.rcParams.update({'figure.max_open_warning': 0})

    # Indicates when cases are above 1e-13
    index = list(i_countries.columns).index(cutoff[country])

    # Set t (actual dates) and time (# of days)
    t = i_countries.columns[index:]
    time = np.linspace(0, len(t), len(t))

    # Country data
    country_infected = i_countries[i_countries.index == country].iloc[0, index:]
    country_recovered = r_countries[r_countries.index == country].iloc[0, index:]
    
    # print('test')
    # Run models
    mse_model = optimize.minimize(rmse, x0, method = 'Nelder-Mead', tol = 1e-7)
    
    return mse_model.x

def graph_scipy(country, i_countries, r_countries, b, g, cutoff):
    """ This function graphs the SIR model predictions and actual values for
        % of pop. infected / recovered for a country between a given time period

        Arguments
        ---------
        country (str): name of country
        i_countries (pd.DataFrame): DataFrame w/ % of pop. infected for countries
        r_countries (pd.DataFrame): DataFrame w/ % of pop. recovered for countries
        b, g (float): variables for SIR model
        cutoff (dict): index is country name and value is date country infected exceeds e^-13
    """

    index = cutoff[country]
    # Generate next 30 days
    # dates = make_dates()
    
    # Country data
    country_infected = i_countries[i_countries.index == country].loc[:, index:]
    country_recovered = r_countries[r_countries.index == country].loc[:, index:]
    
    t_current = list(country_infected.columns)
    time = np.linspace(0, len(t_current), len(t_current))

    # I Graph
    fig = plt.figure(facecolor='w', figsize = (12, 8))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True) # Change from axis_bgcolor
    
    # Graph Info
    ax.set_title(country + ' SIR Model Predictions: Infected')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Infected (% of population)')
    
    # Graph Models
    S, I, R = scipy_model(b, g, time)
    ax.plot(t_current, I, lw=2, label='MSE Infected', marker = '^', ls = 'None') 
    
    # Graph Data
    ax.plot(t_current, country_infected.values[0], lw=2, label='Infected Data', marker = 'x', ls = 'None')

    # Other Graph Info

    # Set x-ticks
    ticks = [x * 7 for x in range(len(t_current) // 7)]
    ticks.append(len(t_current))
    labels = [t_current[7 *n] for n in range(len(t_current) // 7)]
    labels.append(t_current[-1])
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels = labels, rotation = 45)
    
    # Set legend
    legend = ax.legend()
    legend.get_frame().set_alpha(0.8)

    # R Graph
    fig = plt.figure(facecolor='w', figsize = (12, 8))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True) # Change from axis_bgcolor
    
    # Graph Info
    ax.set_title(country + ' SIR Model Predictions: Recovered')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Recovered (% of population)')

    # Graph Models
    ax.plot(t_current, R, lw=2, label='MSE Recovered', marker = '^', ls = 'None')
     
    # Graph Data
    ax.plot(t_current, country_recovered.values[0], lw=2, label='Recovered Data', marker = 'x', ls = 'None')

    # Set x-ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels = labels, rotation = 45)

    # Legend Stuff
    legend = ax.legend()
    legend.get_frame().set_alpha(0.8)
    
    plt.show()

"""
Old
def predict_scipy(country, i_countries, r_countries, b, g, cutoff, dates):

    index = cutoff[country]
    # Generate next 30 days
    dates = make_dates()
    
    # Country data
    country_infected = i_countries[i_countries.index == country].loc[:, index:]
    country_recovered = r_countries[r_countries.index == country].loc[:, index:]
    
    t_current = list(country_infected.columns)
    t_predict = list(country_infected.columns)
    t_predict.extend(dates)
    time = np.linspace(0, len(t_predict), len(t_predict))

    # I Graph
    fig = plt.figure(facecolor='w', figsize = (12, 8))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True) # Change from axis_bgcolor
    
    # Graph Info
    ax.set_title(country + ' SIR Model Predictions: Infected')
    ax.set_xlabel('Time / days')
    ax.set_ylabel('Infected (% of population)')
    
    # Graph Models
    S, I, R = scipy_model(b, g, time)
    ax.plot(t_predict, I, lw=2, label='MSE Infected', marker = '^', ls = 'None') 
    
    # Graph Data
    ax.plot(t_current, country_infected.values[0], lw=2, label='Infected Data', marker = 'x', ls = 'None')

    # Other Graph Info

    # Set x-ticks
    ticks = [x * 7 for x in range(len(t_predict) // 7)]
    ticks.append(len(t_predict))
    labels = [t_predict[7 *n] for n in range(len(t_predict) // 7)]
    labels.append(t_predict[-1])
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels = labels, rotation = 45)
    
    # Set legend
    legend = ax.legend()
    legend.get_frame().set_alpha(0.8)

    # R Graph
    fig = plt.figure(facecolor='w', figsize = (12, 8))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True) # Change from axis_bgcolor
    
    # Graph Info
    ax.set_title(country + ' SIR Model Predictions: Recovered')
    ax.set_xlabel('Time / days')
    ax.set_ylabel('Recovered (% of population)')

    # Graph Models
    ax.plot(t_predict, R, lw=2, label='MSE Recovered', marker = '^', ls = 'None')
     
    # Graph Data
    ax.plot(t_current, country_recovered.values[0], lw=2, label='Recovered Data', marker = 'x', ls = 'None')

    # Set x-ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels = labels, rotation = 45)

    # Legend Stuff
    legend = ax.legend()
    legend.get_frame().set_alpha(0.8)
    
    plt.show()
"""