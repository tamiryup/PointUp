# CSV's Explanatory README

### nyc_sales_original.csv:
This file has all the house categories (i.e. including theaters asylums etc).  
This file does not contain null values (zeros) in the LAND SQUARE FEET, GROSS SQUARE FEET and SALE PRICE columns

### nyc_sales_1.0.csv:
This file contains only the relevant buildings in house category (homes and apartments).  
In addition it contains a new column YEAR TYPE which is a categorization of the building year.

### nyc_sales_1.1.csv:
This is nyc_sales_1.0 after removing outliers (SALE PRICE > 4,000,000 and such).
