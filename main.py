import src.acquistion as aq
import src.processing as prc
import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

CSV_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point?Time=LST&parameters=T2M,RH2M,PRECTOTCORR,PS,WS50M&community=RE&longitude=-0.1677&latitude=51.4627&start=20180101&end=20230101&format=CSV"
CSV_FILE = "weather_data.csv"

data = aq.get_weather_data(CSV_URL,CSV_FILE)
data = data[13:]

#create a new csv file to write the updated data into it
file = open('weather_data_updated.csv', 'w', newline ='')
# writing the data into the new file
with file:  
    write = csv.writer(file)
    write.writerows(data)

#convert the updated csv file into a pandas dataframe
df = pd.read_csv("weather_data_updated.csv")
print(df)

df['datetime'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
df.set_index('datetime', inplace=True)
df.index = pd.to_datetime(df.index)
df = df.drop(columns=['YEAR','MO','DY','HR'])
df = df.rename(columns={'T2M':'Temperature','RH2M':'Humidity','PRECTOTCORR':'Precipitation','PS':'Pressure','WS50M':'Wind Speed'})
print(df)

print(df.isnull().sum())

#outliers
#visual
prc.get_boxplot(df['Temperature'],"Temperature in Celsius", "box_plot_temp")
prc.get_boxplot(df['Humidity'],"Relative Humidity", "box_plot_hum")
prc.get_boxplot(df['Precipitation'],"Precipitation in mm/hour", "box_plot_precip")
prc.get_boxplot(df['Pressure'],"Pressure in kPa", "box_plot_pressure")
prc.get_boxplot(df['Wind Speed'],"Wind Speed in m/s", "box_plot_wind")

contains_negative_1000 = (df == -999).sum()

print(contains_negative_1000)

prc.get_scatter_plot(df,'Temperature','Temperture in Celsius','scatter_temp')
prc.get_scatter_plot(df,'Humidity','Relative Humidity','scatter_hum')
prc.get_scatter_plot(df,'Precipitation','Precipitation in mm/hour','scatter_prec')
prc.get_scatter_plot(df,'Pressure','Pressure in kPa','scatter_pres')
prc.get_scatter_plot(df,'Wind Speed','Wind Speed in m/s','scatter_wind')

bad_data_loc = np.where(df < -500)
print(bad_data_loc)

print(df.index)