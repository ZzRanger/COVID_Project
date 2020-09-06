# Import Libraries
import pandas as pd
import numpy as np
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

""" This module runs the graph code for LSTM.ipynb """
def eda_graph(covid_model_values, ax):
  """ This function creates a figure subplot for global COVID case data

      Arguments
      ---------
      covid_model_values (pd.Series): time series data
      ax (matplotlib.axes._subplots): blank AxesSubplot object

      Returns
      -------
      ax (matplotlib.axes._subplots): AxesSubplot object
  """
  t = [i for i in range(len(covid_model_values))]
  ax.plot(t, covid_model_values)

  tick_loc = [i * 7 for i in range(len(t) // 7)]
  tick_loc.append(len(t))
  tick_label = [covid_model_values.index[7 * i] for i in range(len(t) // 7)]
  tick_label.append(covid_model_values.index[-1])

  ax.set_xticks(tick_loc)
  ax.set_xticklabels(tick_label, rotation = 45)
  return ax

def model_graph(ax, label, predictions, covid_testing_values):
    """ This function plots the LSTM Model Predictions of Global
        COVID Cases in May

    Arguments
    ---------
    ax (matplotlib.axes._subplots): blank AxesSubplot object
    label (str): type of graph
      'baseline': Baseline LSTM Model Predictions
      'tuned': Tuned LSTM Model Predictions
      'compare': Avg. LSTM Model Predictions for Baseline & Tuned Models
    predictions (pd.DataFrame): DataFrame containing model predictions
    covid_testing_values (pd.Series): time series for global COVID cases in May

    Returns
    -------
    ax (matplotlib.axes._subplots): AxesSubplot object
    """
    # x ticks
    tick_loc = [i * 7 for i in range(5)]
    tick_label = [predictions.columns[7 * i] for i in range(5)]
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(tick_label, rotation = 45)

    t = np.linspace(0, len(predictions.columns), len(predictions.columns))

    if label == 'baseline':
      for i in range(10):
          ax.plot(t, predictions.iloc[i, :], label = "Simulation #: " + str(i + 1), alpha = 0.5, marker = 'o', linestyle = 'None')
      ax.plot(t, covid_testing_values, label = 'Actual')

    elif label == 'tuned':
      for i in range(10):
          ax.plot(t, predictions.iloc[i + 10, :], label = "Simulation #: " + str(i + 1), alpha = 0.5, marker = 'o', linestyle = 'None')
      ax.plot(t, covid_testing_values, label = 'Actual')

    elif label == 'compare':
      base = predictions.iloc[0:10, :].sum() / 10
      tune = predictions.iloc[10:19, :].sum() / 10
      ax.plot(t, base, label = 'Base Model Avg.')
      ax.plot(t, tune, label = 'Tuned Model Avg.')
      ax.plot(t, covid_testing_values, label = 'Actual')

    ax.legend()
    return ax
