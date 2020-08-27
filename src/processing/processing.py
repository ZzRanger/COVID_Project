# Import libraries
from pandas import read_csv
import pandas as pd
from pandas import DataFrame as df

def import_data(confirmed, deaths, recovered): 
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
    # Import World Population as Dictionary
    import_country_pop = read_csv(raw_country_pop)
    country_pop = dict()

    # Population in millions
    for i in range(len(import_country_pop.index)):
        country_pop[import_country_pop.iloc[i, 0]] = float(import_country_pop.iloc[i,1]) / 1000
    return country_pop