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


    
df_theaters = pd.read_csv("theaters.csv")
df_deals = pd.read_csv("new_hc_mus.csv")

"""cleaning a bit"""
df_deals = df_deals.dropna(subset=['lat','lng'])

"""adding rows for park location + acres"""
df_deals["theater_dist"] = np.nan
df_deals["theater_in_brodway"] = np.nan
df_deals["theater_lat"] = np.nan
df_deals["theater_lng"] = np.nan
df_deals["theaters_cnt_500"] = np.nan
df_deals["theaters_cnt_1000"] = np.nan
df_deals["theaters_cnt_3000"] = np.nan
df_deals["broadway_cnt_500"] = np.nan
df_deals["broadway_cnt_1000"] = np.nan
df_deals["broadway_cnt_3000"] = np.nan

i = 0


for d_cnt in range(df_deals.shape[0]):

    if d_cnt%1000 == 0:
        print(d_cnt)

    try:
        lat = df_deals["lat"][d_cnt]
        lng = df_deals["lng"][d_cnt]
    except:
        print("error in line {0}".format(d_cnt))
        continue

    #if d_cnt == 100:
    #    break

    theater_info = df_theaters["the_geom"][0].split(" ")
    theater_info = theater_info[1:]

    """theater + broadway counter"""
    t_cnt500 = 0
    t_cnt1000 = 0
    t_cnt3000 = 0
    broadway_cnt500 = 0
    broadway_cnt1000 = 0
    broadway_cnt3000 = 0

    min_dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],float(theater_info[1]),float(theater_info[0]))
    min_dist_lat = theater_info[0]
    min_dist_lng = theater_info[1]
    
    for t_cnt in range(df_theaters.shape[0]):
        theater_info = df_theaters["the_geom"][t_cnt].split(" ")
        theater_info = theater_info[1:]
        dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],float(theater_info[1]),float(theater_info[0]))
        if dist<min_dist:
            min_dist = dist
            in_broadway = 1 if df_theaters["ADDRESS1"][t_cnt].lower().count("broadway")>0 else 0
            min_dist_lat = theater_info[0]
            min_dist_lng = theater_info[1]
        if dist < 0.5:
            t_cnt500 = t_cnt500 + 1
            if df_theaters["ADDRESS1"][t_cnt].lower().count("broadway")>0:
                broadway_cnt500 + 1
        if dist < 1.0 and dist > 0.5:
            t_cnt1000 = t_cnt1000 + 1
            if df_theaters["ADDRESS1"][t_cnt].lower().count("broadway")>0:
                broadway_cnt1000 = broadway_cnt1000 + 1
        if dist < 3.0 and dist > 1.0:
            t_cnt3000 = t_cnt3000 + 1
            if df_theaters["ADDRESS1"][t_cnt].lower().count("broadway")>0:
                broadway_cnt3000 = broadway_cnt3000 + 1
            
    df_deals.at[d_cnt,"theater_dist"] = min_dist
    df_deals.at[d_cnt,"theater_in_brodway"] = in_broadway
    df_deals.at[d_cnt,"theater_lat"] = min_dist_lat
    df_deals.at[d_cnt,"theater_lng"] = min_dist_lng

    df_deals.at[d_cnt,"theaters_cnt_500"] = t_cnt500
    df_deals.at[d_cnt,"theaters_cnt_1000"] = t_cnt1000
    df_deals.at[d_cnt,"theaters_cnt_3000"] = t_cnt3000
    df_deals.at[d_cnt,"broadway_cnt_500"] = broadway_cnt500
    df_deals.at[d_cnt,"broadway_cnt_1000"] = broadway_cnt1000
    df_deals.at[d_cnt,"broadway_cnt_3000"] = broadway_cnt3000
        
(df_deals["theater_dist"].head())
df_deals.to_csv("new_hc_mus_theaters.csv")
    
    
            
    
    
