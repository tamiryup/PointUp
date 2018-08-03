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


    
df_parks = pd.read_csv("parks.csv")
df_deals = pd.read_csv("new_hc_mus_theaters.csv")

"""cleaning a bit"""
df_deals = df_deals.dropna(subset=['lat','lng'])
df_parks = df_parks[df_parks["lat"]!=0]

"""adding rows for park location + acres"""
df_deals["park_dist"] = np.nan
df_deals["park_acres"] = np.nan
df_deals["park_lat"] = np.nan
df_deals["park_lng"] = np.nan
df_deals["park_cnt_200"] = np.nan
df_deals["park_cnt_1000"] = np.nan
df_deals["park_cnt_3000"] = np.nan
df_deals["acres_cnt_200"] = np.nan
df_deals["acres_cnt_1000"] = np.nan
df_deals["acres_cnt_3000"] = np.nan

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

    min_dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],df_parks["lat"][0],df_parks["lng"][0])
    min_dist_acres = df_parks["ACRES"][0]
    min_dist_lat = df_parks["lat"][0]
    min_dist_lng = df_parks["lng"][0]

    """park counters"""
    p_cnt_200 = 0
    p_cnt_1000 = 0
    p_cnt_3000 = 0

    """acres acounter"""
    a_cnt_200 = 0.0
    a_cnt_1000 = 0.0
    a_cnt_3000 = 0.0
    
    for park_acres,park_lat,park_lng in zip(df_parks["ACRES"],df_parks["lat"],df_parks["lng"]):
        dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],park_lat,park_lng)
        if dist<min_dist:
            min_dist = dist
            min_dist_acres = park_acres
            min_dist_lat = park_lat
            min_dist_lng = park_lng

        if dist <= 0.2:
            p_cnt_200 = p_cnt_200 + 1
            a_cnt_200 = a_cnt_200 + park_acres

        if dist <= 1 and dist > 0.2:
            p_cnt_1000 = p_cnt_1000 + 1
            a_cnt_1000 = a_cnt_1000 + park_acres

        if dist <= 3 and dist > 1:
            p_cnt_3000 = p_cnt_3000 + 1
            a_cnt_3000 = a_cnt_3000 + park_acres
        
    df_deals.at[d_cnt,"park_dist"] = min_dist
    df_deals.at[d_cnt,"park_acres"] = min_dist_acres
    df_deals.at[d_cnt,"park_lat"] = min_dist_lat
    df_deals.at[d_cnt,"park_lng"] = min_dist_lng
    df_deals.at[d_cnt,"park_cnt_200"] = p_cnt_200
    df_deals.at[d_cnt,"park_cnt_1000"] = p_cnt_1000
    df_deals.at[d_cnt,"park_cnt_3000"] = p_cnt_3000
    df_deals.at[d_cnt,"acres_cnt_200"] = a_cnt_200
    df_deals.at[d_cnt,"acres_cnt_1000"] = a_cnt_1000
    df_deals.at[d_cnt,"acres_cnt_3000"] = a_cnt_3000   
        
print(df_deals["park_dist"].head())
df_deals.to_csv("new_hc_mus_theaters_parks.csv")
    
    
            
    
    
