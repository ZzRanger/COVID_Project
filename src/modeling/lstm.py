""" This module processes the data for LSTM.ipynb """
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

# Process Data For Model
def split_sequence(sequence, n_steps):
	""" This function splits data into input (X) and output (y) sets

		Arguments
		---------
		sequence (pd.Series): time series data
		n_steps (int): ratio of X to y

		Returns
		-------
		X, y (np.ndarray): X and y values for model
	"""

	X, y = list(), list()
	for i in range(len(sequence)):
			# find the end of this pattern
			end_ix = i + n_steps
			# check if we are beyond the sequence
			if end_ix > len(sequence)-1:
					break
			# gather input and output parts of the pattern
			seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
			X.append(seq_x)
			y.append(seq_y)
	return asarray(X), asarray(y)
		
def model_data(covid_model_values):
	""" This function splits data into training and validation sets

		Arguments
		---------
		covid_model_values (pd.Series): time series data

		Returns
		-------
		X_train, X_valid, y_train, y_valid (np.ndarray): training and validation sets
	"""
	# specify the window size
	n_steps = 2
	# split into samples
	X, y = split_sequence(covid_model_values, n_steps) 
	# reshape into [samples, timesteps, features]
	X = X.reshape((X.shape[0], X.shape[1], 1))
	# split into train/test
	n_test = 8
	X_train, X_valid, y_train, y_valid = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
	return X_train, X_valid, y_train, y_valid
