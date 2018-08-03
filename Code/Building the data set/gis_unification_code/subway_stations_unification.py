from math import sin, cos, sqrt, atan2, radians
import pandas as pd
import numpy as np


def calc_dist(lat1,lon1,lat2,lon2):
    """recieve two lat,lon = latitude,longditue pairs. clacs the walking distance"""
    R = 6373.0 #[km]
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


df_stations = pd.read_csv("stations_coordinates.csv")
df_deals = pd.read_csv("new_hc_mus_theaters.csv")

"""cleaning a bit"""
df_deals = df_deals.dropna(subset=['lat','lng'])
df_stations = df_stations[df_stations["lat"]!=0]

"""adding rows for park location + acres"""
df_deals["station_dist"] = np.nan
df_deals["station_cnt_500"] = np.nan
df_deals["station_cnt_1000"] = np.nan
df_deals["station_cnt_3000"] = np.nan

i = 0 

for d_cnt in range(df_deals.shape[0]):

    if d_cnt % 1000 == 0:
        print(d_cnt)

    try:
        lat = df_deals["lat"][d_cnt]
        lng = df_deals["lng"][d_cnt]
    except:
        print("error in line {0}".format(d_cnt))
        continue
    
    #if d_cnt == 100:
    #    break

    min_dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],df_stations["lat"][0],df_stations["lng"][0])
    min_dist_lat = df_stations["lat"][0]
    min_dist_lng = df_stations["lng"][0]

    """station counters"""
    s_cnt_500 = 0
    s_cnt_1000 = 0
    s_cnt_3000 = 0

    
    for s_lat,s_lng in zip(df_stations["lat"],df_stations["lng"]):
        dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],s_lat,s_lng)
        if dist<min_dist:
            min_dist = dist
            min_dist_lat = s_lat
            min_dist_lng = s_lng

        if dist <= 0.5:
            s_cnt_500 = s_cnt_500 + 1

        if dist <= 1 and dist > 0.5:
            s_cnt_1000 = s_cnt_1000 + 1

        if dist <= 3 and dist > 1:
            s_cnt_3000 = s_cnt_3000 + 1
        
    df_deals.at[d_cnt,"station_dist"] = min_dist
    df_deals.at[d_cnt,"station_cnt_500"] = s_cnt_500
    df_deals.at[d_cnt,"station_cnt_1000"] = s_cnt_1000
    df_deals.at[d_cnt,"station_cnt_3000"] = s_cnt_3000
        
print(df_deals["station_dist"].head())
df_deals.to_csv("out.csv")
    
    
