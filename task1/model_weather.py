import datetime

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.parser import parse

#pm2.5 폴더 경로
# dir_path = '/volante/data/GEOS-Chem/ExtData/GEOS_0.25x0.3125_CH/GEOS_FP/'
dir_path = '/veloce/jij/ExtData/GEOS_0.25x0.3125_CH/GEOS_FP/'
dataset = []

def saveModelData(dir_path,period,variables) :
    #시간 나누기
    s_period = period.split("-")
    begin = parse(s_period[0])
    end = parse(s_period[1])

    #변수 나누기
    s_vars = variables.split(",")

    while begin <= end :
        strDate = begin.strftime("%Y-%m-%d").split('-')
        path = dir_path + strDate[0] + '/' + strDate[1] + '/' + 'GEOSFP.' + strDate[0] + \
        strDate[1] +  strDate[2] +'.A1.025x03125.CH.nc'
        ds = xr.open_dataset(path)

        choose_vars=[]
        for var in s_vars :
            choose_vars.append(ds[var])

        dataset.append(choose_vars)
        begin += datetime.timedelta(days=1)

    return s_vars

def get_weather(data, variables, lat, lon) :
    day = 0
    loc_data = pd.DataFrame()

    while day < len(data):
        data_1d = pd.DataFrame(pd.Series(data[day][0]['time'],name='time'))
        for i in range (0,len(variables)) :
            var = pd.Series(np.array(data[day][i][:, lat, lon]),name=variables[i])
            data_1d=pd.concat([data_1d,var],axis=1)

        if day==0 :
            loc_data = data_1d
        else :
            loc_data = loc_data.append(data_1d)

        day += 1

    loc_data.set_index('time',inplace=True)
    return loc_data

def run(period,variables) :
    variable = saveModelData(dir_path,period,variables)
    return dataset, variable

