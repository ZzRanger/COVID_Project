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