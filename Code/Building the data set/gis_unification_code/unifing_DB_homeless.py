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


    
df_homeless_center = pd.read_csv("Directory_Of_Homeless_Drop-_In_Centers.csv")
df_deals = pd.read_csv("nyc_sales_coor_vik.csv")

"""cleaning a bit"""
df_deals = df_deals.dropna(subset=['lat','lng'])
df_deals['lat'] = df_deals.loc[:,'lat'].round(7)
df_deals['lng'] = df_deals.loc[:,'lng'].round(7)

"""adding rows for park location + acres"""
df_deals["homeless_center_dist"] = np.nan
df_deals["homeless_center_lat"] = np.nan
df_deals["homeless_center_lng"] = np.nan

print(df_deals.shape)

print(df_deals.loc[0,"lat"])
print(df_deals.loc[69993,"lat"])

df_deals.to_csv("check.csv")

i = 0 

for d_cnt in range(df_deals.shape[0]):

    if d_cnt % 10000 == 0:
       print(d_cnt)

    try:
        lat = df_deals["lat"][d_cnt]
        lng = df_deals["lng"][d_cnt]
    except:
        print("error in line {0}".format(d_cnt))
        continue

    #if d_cnt == 100:
    #    break

    min_dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],df_homeless_center["Latitude"][0],df_homeless_center["Longitude"][0])
    min_dist_lat = df_homeless_center["Latitude"][0]
    min_dist_lng = df_homeless_center["Longitude"][0]
    
    for i in range(len(df_homeless_center["Latitude"])):
        dist = calc_dist(df_deals["lat"][d_cnt],df_deals["lng"][d_cnt],df_homeless_center["Latitude"][i],df_homeless_center["Longitude"][i])
        if dist<min_dist:
            min_dist = dist
            min_dist_lat = df_homeless_center["Latitude"][i]
            min_dist_lng = df_homeless_center["Longitude"][i]

        
    df_deals.at[d_cnt,"homeless_center_dist"] = min_dist
    df_deals.at[d_cnt,"homeless_center_lat"] = min_dist_lat
    df_deals.at[d_cnt,"homeless_center_lng"] = min_dist_lng  
        
print(df_deals["homeless_center_dist"].head())
df_deals.to_csv("new_hc.csv")
    
    
            
    
    
