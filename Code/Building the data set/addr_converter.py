###### used to convert the address in the orignal dataset to the format needed for google api ######

file = open("nyc_sales_binned.csv","r")
cont = file.read()
deal_arr = cont.split("\n")
deal_arr = deal_arr[1:] # get rid of the title
addr_file = open("adresses.csv","w")


#Functions Area -

addr_dict = {"Road": "Rd","road":"Rd","ROAD":"Rd", "RD":"Rd","ST.":"St","STREET":"St", "AVENUE":"Ave", "AVE":"Ave", "Rd":"Rd","St":"St","Ave":"Ave", "DRIVE":"Dr","DR":"Dr","Dr":"Dr","LANE":"Ln","LN":"Ln","LOOP":"Loop" ,"COURT":"Ct", "CT":"Ct","PLACE":"Pl","PL":"Pl","BOULEVARD":"Blvd","BLVD":"Blvd","WAY":"Way"}

"""gets a street description and fixes it according to the dictionary above"""
def fix_street_description(desc):
    return addr_dict[desc]

"""gets array of strings and clears for blank"""
def clean_arr_for_blank(arr): #TEST SUCCESSFUL
    return list(filter(lambda x: x!='',arr))
  
"""gets a street name in array format, and fixes to Capital letters in the begining of each word and lowercase in the rest"""
def fix_street_name(street_name): #TEST SUCCESSFUL
    #street_name = clean_arr_for_blank(street_name)
    #street_name = street_name[:-1]
    for i,val in enumerate(street_name):
        new_val = val[0].upper()
        val = val[1:]
        for c in val:
            new_val = new_val + c.lower()
        street_name[i] = new_val
    return (' ').join(street_name)
            

"""get addr string, and fixes it according to the geocode format"""
def fix_addr(addr): #WRITING
    addr_arr = addr.split(' ')
    addr_arr = clean_arr_for_blank(addr_arr)
    if addr_arr[-1] not in addr_dict:
        addr_arr = addr_arr + ["St"]
    new_addr = fix_street_name(addr_arr[0:len(addr_arr)-1]) + " " + fix_street_description(addr_arr[-1])
    return new_addr
    
    

#################################

print(len(deal_arr)-1)

for cnt,row in enumerate(deal_arr):
    deal = row.split(',')
    addr_file.write("{0},{1},{2},{3},{4}\n".format(deal[0], fix_addr(deal[8]), "New York", "NY",deal[10]))
    print(cnt)
    if cnt == len(deal_arr)-2:
        break






file.close()
addr_file.close()
