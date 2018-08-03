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


    
df_museums = pd.read_csv("museums.csv")
df_deals = pd.read_csv("new_hc.csv")

"""cleaning a bit"""
df_deals = df_deals.dropna(subset=['lat','lng'])

"""adding rows for park location + acres"""
df_deals["museum_dist"] = np.nan
df_deals["museum_lat"] = np.nan
df_deals["museum_lng"] = np.nan
df_deals["museum_cnt_500"] = np.nan
df_deals["museum_cnt_1000"] = np.nan
df_deals["museum_cnt_3000"] = np.nan

i = 0

print(df_deals.shape)

for d_cnt in range(df_deals.shape[0]):

    if d_cnt % 1000 == 0:
        print(d_cnt)

    try:
        lat = df_deals["lat"][d_cnt]
        lng = df_deals["lng"][d_cnt]
    except:
        print("error in line {0}".format(d_cnt))
        continue

##    if d_cnt >= 6530:
##        print(d_cnt)
##        print(df_deals["lat"][d_cnt])
##        print(df_deals["lng"][d_cnt])
##        #break

    first_museum_coor = df_museums["the_geom"][0].split(" ")

    min_dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],float(first_museum_coor[1]),float(first_museum_coor[0]))
    min_dist_lat = first_museum_coor[1]
    min_dist_lng = first_museum_coor[0]

    """museum counters"""
    m_cnt_500 = 0
    m_cnt_1000 = 0
    m_cnt_3000 = 0

    for i in range(len(df_museums["the_geom"])):
        mus_coor = df_museums["the_geom"][i].split(" ")
        dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],float(mus_coor[1]),float(mus_coor[0]))
        if dist<min_dist:
            min_dist = dist
            min_dist_lat = mus_coor[1]
            min_dist_lng = mus_coor[0]

        if dist <= 0.5:
            m_cnt_500 = m_cnt_500 + 1

        if dist <= 1 and dist > 0.2:
            m_cnt_1000 = m_cnt_1000 + 1

        if dist <= 3 and dist > 1:
            m_cnt_3000 = m_cnt_3000 + 1
        
    df_deals.at[d_cnt,"museum_dist"] = min_dist
    df_deals.at[d_cnt,"museum_lat"] = min_dist_lat
    df_deals.at[d_cnt,"museum_lng"] = min_dist_lng
    df_deals.at[d_cnt,"museum_cnt_500"] = m_cnt_500
    df_deals.at[d_cnt,"museum_cnt_1000"] = m_cnt_1000
    df_deals.at[d_cnt,"museum_cnt_3000"] = m_cnt_3000  
        
print(df_deals["museum_dist"].head())
df_deals.to_csv("new_hc_mus.csv")
    
    
            
    
    
