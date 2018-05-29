import pandas as pd

areas = ["bronx", "brooklyn", "manhattan", "queens", "statenisland"]
frames = []

def append_rows(df, target, cols = False):
    length = len(target.index)
    df.columns = target.columns
    df.drop(df.index[:5], inplace=True)
    target = target.append(df)
    #target.iloc[length:]=df.iloc[3:]
    return target



def main():
    target = pd.DataFrame()
    for year in range(2010,2018):
        for area in areas:
            cols = 0
            print(str(year)+" "+area)
            x = pd.ExcelFile("/Users/tamir/Downloads/"+ str(year)+"_"+area+".xls")
            df = x.parse()
            if area=="bronx" and year==2010:
                target = pd.DataFrame(columns=df.iloc[2])
            target = append_rows(df,target,cols)
            #target  = append_rows(df,target,cols)
    target.to_csv("/Users/tamir/Downloads/nyc_sales.csv", index=False)
    print(target.shape)

main()


