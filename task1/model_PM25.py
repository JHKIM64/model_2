import datetime

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.parser import parse

#pm2.5 폴더 경로
dir_path = '/veloce2/sel/GC13.0.0/OutputDir/geosfp_0.25x0.3125_0203/'

dataset = []

def saveModelData(dir_path,period) :
    s_period = period.split("-")
    begin = parse(s_period[0])
    end = parse(s_period[1])
    while begin <= end :
        strDate = begin.strftime("%Y%m%d")
        path = dir_path + 'GEOSChem.AerosolMass.' + strDate +'_0000z.nc4'
        ds = xr.open_dataset(path)
        dataset.append(ds['PM25'])
        begin += datetime.timedelta(days=1)

def get_pm25(data, level, lat, lon) :
    day = 0
    loc_data = pd.DataFrame()

    while day < data.__len__() :
        data_1d_mod = data[day][:, level, lat, lon]
        time_1d=np.array(data_1d_mod['time'])
        pm25_1d=np.array(data_1d_mod)

        if day==0 :
            loc_data = pd.DataFrame(pm25_1d,time_1d)
        else :
            loc_data = loc_data.append(pd.DataFrame(pm25_1d,time_1d))

        day += 1

    loc_data.columns = ['PM25']
    loc_data.index.name = 'time'
    return loc_data

def run(period) :
    saveModelData(dir_path,period)
    return dataset



