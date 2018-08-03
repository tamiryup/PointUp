###### using google api to geocode the addresses of the deals (address to coordinates) ######

import json
import requests

START = 0
END = 190000

json_results = open("json_results_{0}_{1}.csv".format(START,END),"w")
loca_results = open("loca_results_{0}_{1}.csv".format(START,END),"w")
addr_file = open("adresses.csv","r")

cont = addr_file.read()
addr_arr = cont.split("\n")

addr_arr = addr_arr[START:END]

for i,row in enumerate(addr_arr):
    row = row.split(",")
    response = requests.get("https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyC2eFzz87JAwg-iU7Xr5V3Z0KoEAsPTGrE&address={0}".format(",".join(row[1:])))
    data = response.json()
    lat = "{0}".format(((((data["results"])[0])["geometry"])["location"])["lat"])
    lng = "{0}".format(((((data["results"])[0])["geometry"])["location"])["lng"])
    loca_results.write(",".join(row + [lat] + [lng] + ["\n"]))
    json_results.write(json.dumps(data) + "\n")
    if i%10 == 0:
        print(i+START)

json_results.close()
loca_results.close()
addr_file.close()
