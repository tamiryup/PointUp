import pandas as pd
import numpy as np


"""
this condition leaves only apartment buildings and family homes
"""
def cond(input):
    return "APARTMENTS" in input or "FAMILY" in input


def main():
    df = pd.read_csv("/Users/tamir/Documents/PointUp/nyc_sales_data/nyc_sales_1.0.csv")
    df = df[df["BUILDING CLASS CATEGORY"].map(cond)]
    print(df.shape)
    print(df["BUILDING CLASS CATEGORY"].unique())
    df.to_csv("/Users/tamir/Downloads/nyc_sales_homes.csv", index=False)

main()