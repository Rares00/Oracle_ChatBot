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


bad_sensor_val_loc = df[df < -500].dropna()

df_updated = df[~(df < -500).any(axis=1)]

print(bad_sensor_val_loc)
print (df)
print(df_updated)
print(df.index)

df_updated = df_updated.resample('H').interpolate()
print(df_updated.shape)

# plot and compare them


prc.compare_graph(df.index,df['Temperature'],bad_sensor_val_loc.index,bad_sensor_val_loc['Temperature'],df_updated.index,df_updated['Temperature'],'Date Time','Temperature','Comparison Plot','comp_temp')
prc.compare_graph(df.index,df['Humidity'],bad_sensor_val_loc.index,bad_sensor_val_loc['Humidity'],df_updated.index,df_updated['Humidity'],'Date Time','Humidity','Comparison Plot','comp_hum')
prc.compare_graph(df.index,df['Precipitation'],bad_sensor_val_loc.index,bad_sensor_val_loc['Precipitation'],df_updated.index,df_updated['Precipitation'],'Date Time','Precipitation','Comparison Plot','comp_prec')
prc.compare_graph(df.index,df['Pressure'],bad_sensor_val_loc.index,bad_sensor_val_loc['Pressure'],df_updated.index,df_updated['Pressure'],'Date Time','Pressure','Comparison Plot','comp_pres')
prc.compare_graph(df.index,df['Wind Speed'],bad_sensor_val_loc.index,bad_sensor_val_loc['Wind Speed'],df_updated.index,df_updated['Wind Speed'],'Date Time','Wind Speed','Comparison Plot','comp_ws')
plt.close()
plt.clf()
#boxplot for new dataset

prc.get_boxplot(df_updated['Temperature'],"Temperature in Celsius", "box_plot_temp_new")
prc.get_boxplot(df_updated['Humidity'],"Relative Humidity", "box_plot_hum_new")
prc.get_boxplot(df_updated['Precipitation'],"Precipitation in mm/hour", "box_plot_precip_new")
prc.get_boxplot(df_updated['Pressure'],"Pressure in kPa", "box_plot_pressure_new")
prc.get_boxplot(df_updated['Wind Speed'],"Wind Speed in m/s", "box_plot_wind_new")

#new scatter plots

prc.get_scatter_plot(df_updated,'Temperature','Temperture in Celsius','scatter_temp_new')
prc.get_scatter_plot(df_updated,'Humidity','Relative Humidity','scatter_hum_new')
prc.get_scatter_plot(df_updated,'Precipitation','Precipitation in mm/hour','scatter_prec_new')
prc.get_scatter_plot(df_updated,'Pressure','Pressure in kPa','scatter_pres_new')
prc.get_scatter_plot(df_updated,'Wind Speed','Wind Speed in m/s','scatter_wind_new')

#next step check for other oultier and cap/give new value