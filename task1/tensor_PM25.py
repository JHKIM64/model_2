import pandas as pd
import model_PM25 as PM25
import model_weather as weather
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def get_one_loc_tensor(lat,lon,period,variables) :
    pm25_data,weather_data,weather_vars=model_run(period, variables)

    latitude = np.array(pm25_data[0]['lat'])
    longitude = np.array(pm25_data[0]['lon'])

    lon_point = (np.abs(longitude - lon)).argmin()
    lat_point = (np.abs(latitude - lat)).argmin()

    pm25_df = PM25.get_pm25(pm25_data, 0, lat_point, lon_point)
    weather_df = weather.get_weather(weather_data,weather_vars,lat_point,lon_point)

    before_tensor = pd.concat([pm25_df,weather_df],axis=1)

    print("plot data?  y / n ")
    if sys.stdin.readline().strip() == 'y' :
        correlation(before_tensor)

    return before_tensor

def model_run(period, variables) :
    pm25_data = PM25.run(period)
    weather_data, weather_vars = weather.run(period, variables)
    return pm25_data,weather_data,weather_vars

def correlation(df) :
    df.plot(subplots=True)
    plt.savefig('./model_data.png')
    corr = df.corr()
    sns.clustermap(corr,annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
    plt.savefig('./model_data_corr.png')

# get_one_loc_tensor(37.26,125.57,"20200201-20200229", "TS,U10M,V10M")