""" This module runs the model and graph code for Gaussian.ipynb """

# Library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import diff
import math
from scipy.integrate import quad
from scipy.integrate import odeint

def graph_death_rate(total_us_deaths, us_death_rate):
    """ This function graphs the # of deaths & death 
        rate in the US between a given time period

        Arguments
        ---------
        total_us_deaths (pd.Series): time series with # of US deaths
        us_death_rate (pd.Series): time series with US death rate

        Returns
        -------
        fig (matplotlib.pyplot.figure): graph figure

        Make sure total_us_deaths and us_death_rate have the same length 
    """

    t = np.linspace(0, len(us_death_rate), len(us_death_rate))

    # Graph 
    fig = plt.figure(figsize = (15, 24))

    # Set x labels
    ticks = [i * 7 for i in range(len(t) // 7 + 1)]
    labels = [us_death_rate.index[i] for i in ticks]

    # Graph # of deaths
    ax = fig.add_subplot(211, facecolor = '#ffffff')
    ax.plot(t, total_us_deaths)
    ax.set_title('Total US COVID Deaths')
    ax.set_ylabel('# of Deaths')
    ax.set_xlabel('Dates')

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels = labels, rotation = 45)

    # Graph death rate
    ax = fig.add_subplot(212, facecolor = '#000000')
    ax.plot(t, us_death_rate)
    ax.set_title('Total US COVID Deaths')
    ax.set_ylabel('Daily # of Deaths')
    ax.set_xlabel('Dates')

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels = labels, rotation = 45)

    return fig

# Integral for Gaussian Function
integrand = lambda x: math.exp(-(x**2))

# Gaus Function
def gaus_func(a, b, p, t):
    """ This function inputs the following arguments into the 
        Gaussian Error Function Equation and returns the result.
        
        For more information about the equation, see 
        Background Information in Gaussian.ipynb 

        Arguments
        ---------
        a, b, p (float): variables in equation 
        t (int): time 

        Returns
        -------
        float: y value of Gaussian function
    """
    # Integral Bounds
    lb = 0
    ub = a * (t-b)

    # Integral
    inte = quad(integrand, lb, ub)

    # Equation
    return p/2 * (1 + (2 / math.sqrt(math.pi)) * inte[0])

def d_rate(a, b, p, dur):
    """ This function returns the values of the Gaussian Equation
        from gaus_func between 0 and dur

        For more information about the equation, see 
        Background Information in Gaussian.ipynb 

        Arguments
        ---------
        a, b, p (float): variables in equation 
        dur (int): interval of time 

        Returns
        -------
        list: list of y values of Gaussian function between 0 and dur
    """

    rate = []
    for i in range(dur):
        rate.append(gaus_func(a, b, p, (i + 1)))
    return rate

def graph_error_func(t, rate):
    """ This function graphs the Gaussian Error Function
        
        For more information about the equation, see 
        Background Information in Gaussian.ipynb 

        Arguments
        ---------
        t (numpy.ndarray): x values from 0 to len(rate)
        rate (list): y values of Gaussian Equation from gaus_func

        Returns
        -------
        fig (matplotlib.pyplot.figure): graph figure

        Make sure total_us_deaths and us_death_rate have the same length 
    """
    # Sample graph plot
    fig = plt.figure(facecolor='w', figsize = (12, 8))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True) # Change from axis_bgcolor
    ax.plot(t, rate, 'b', alpha=0.5, lw=2, label='Deaths')
    ax.set_xlabel('Time / days')
    ax.set_ylabel('# of Deaths')
    ax.set_title('Sample Gaussian Error Function')
    # ax.set_ylim(0, 10000000)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.8)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    return fig

def scipy_gaus(total_us_deaths):
    """ This function fits COVID death data to the Gaussian Error Function.
        
        For more information about the equation, see 
        Background Information in Gaussian.ipynb 

        Arguments
        ---------
        a, b, p (float): variables in equation 
        t (int): time 

        Returns
        -------
        val (OptimizeResult): OptimizeResult object from scipy.optimize. 
        See https://bit.ly/32SfMU9 for more info
        dur (int): length of total_us_deaths
    """
    # Cost Function
    def rmse(inpt):
        """ This function is the loss function used by scipy.optimize.minimize

            Arguments
            ---------
            inpt (list): list with the following structure [a, b, p]  
                a, b, and p are variables in Gaussian Error Function
                
                For more information about the equation, see 
                Background Information in Gaussian.ipynb 
            
            Returns
            -------
            float: rmse of Gaussian function
        """
        a = inpt[0]
        b = inpt[1]
        p = inpt[2]
        rate = d_rate(a, b, p, dur)
        sse = 0
        for i in range(dur):
            error_one = abs(rate[i] - total_us_deaths[i])**2
            sse += error_one 
        return math.sqrt(sse / dur)

    # Initial guess
    x0 = [0.2, 2, 100000]

    # Other variables
    dur = len(total_us_deaths)

    # Minimize function
    val = optimize.minimize(rmse, x0, method = 'Nelder-Mead', tol = 0.00000001)
    return val, dur

def scipy_gaus_graph(final_rate, total_us_deaths, us_death_rate):
    """ This function graphs the Gaussian model predictions and actual values 
        for # of deaths & death rate in the US between a given time period

        Arguments
        ---------
        final_rate (list): y values of Gaussian Equation from scipy_gaus
        total_us_deaths (pd.Series): time series with # of US deaths
        us_death_rate (pd.Series): time series with US death rate

        Returns
        -------
        fig (matplotlib.pyplot.figure): graph figure

        Make sure final_rate, total_us_deaths, and us_death_rate have the same length 
    """

    fig = plt.figure(figsize = (12, 16))

    dates = total_us_deaths.index
    t = np.linspace(0, len(dates), len(dates))

    # Set x-ticks
    ticks = [x * 7 for x in range(len(t) // 7 + 1)]
    labels = [dates[n] for n in ticks]

    ax = fig.add_subplot(211, facecolor = '#000000')
    ax.plot(t, final_rate, color = 'white', marker = 'x', label = 'Model')
    # ax.plot(t, variance_rate)
    ax.plot(t, total_us_deaths, color = 'red', label = 'Actual')
    ax.set_title('Total COVID - 19 Deaths in US')
    ax.set_ylabel('Number of deaths')
    ax.set_xlabel('Dates')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation = 45)
    ax.legend()

    # Death rate
    death_rate = diff(final_rate)
    t = np.linspace(0, len(total_us_deaths) - 1, len(total_us_deaths) - 1)

    ax = fig.add_subplot(212, facecolor = '#000000')
    ax.plot(t, death_rate, color = 'white', marker = 'x', label = 'Model')
    ax.plot(t, us_death_rate[1:], color = 'red', label = 'Actual')
    ax.set_title('Rate of COVID - 19 Deaths in US')
    ax.set_ylabel('Rate of Deaths')
    ax.set_xlabel('Dates')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation = 45)
    ax.legend()

"""
Old
def scipy_predict_graph(final_rate, total_us_deaths, us_death_rate, future_dates):

    fig = plt.figure(figsize = (12, 16))

    dates = list(total_us_deaths.index)
    t = np.linspace(0, len(dates), len(dates))
    t_predict = np.linspace(0, len(final_rate), len(final_rate))
    dates.extend(future_dates)
    # Set x-ticks
    ticks = [x * 7 for x in range(len(dates) // 7 + 1)]
    labels = [dates[n] for n in ticks]

    ax = fig.add_subplot(211, facecolor = '#000000')
    ax.plot(t_predict, final_rate, color = 'white', marker = 'x', label = 'Model')
    # ax.plot(t, variance_rate)
    ax.plot(t, total_us_deaths, color = 'red', label = 'Actual')
    ax.set_title('Total COVID - 19 Deaths in US')
    ax.set_ylabel('Number of deaths')
    ax.set_xlabel('Days since e^-15 deaths / day')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation = 45)
    ax.legend()

    # Death rate
    death_rate = diff(final_rate)

    ax = fig.add_subplot(212, facecolor = '#000000')
    ax.plot(t_predict[1:], death_rate, color = 'white', marker = 'x', label = 'Model')
    ax.plot(t, us_death_rate, color = 'red', label = 'Actual')
    ax.set_title('Rate of COVID - 19 Deaths in US')
    ax.set_ylabel('Rate of deaths')
    ax.set_xlabel('Days since e^-15 deaths / day')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation = 45)
    ax.legend()

    return fig
"""
