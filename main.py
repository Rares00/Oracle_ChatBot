import src.acquistion as aq
import src.processing as prc
import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch import nn, optim
import pickle


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

#check for missing values
print(df.isnull().sum())

#outliers
#visualion

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

#deal with the other outliers

#most of the data seems clean at this point. The only one that can still have outliers would be the precipitaion

print(df.nlargest(50,'Precipitation'))

#filter the data for the model:
filter_data = prc.data_filter_ml(df_updated, "2018-01-01","2023-01-01",
                                  ["Temperature", "Humidity", "Precipitation", "Pressure", "Wind Speed"])

print(filter_data)

################## inference ####
df = filter_data.copy()

# load the scaler object from the file
with open('./pickle_files/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./pickle_files/scaler_temp.pkl', 'rb') as f:
    scaler_temp = pickle.load(f)
with open('./pickle_files/scaler_precip.pkl', 'rb') as f:
    scaler_precip = pickle.load(f)

# extract the last 336 data points from df
data = df.iloc[-337:-1, :]  # df = m.filter_data

# normalize the input data using the pre-trained scaler
scaled_data = scaler.transform(data)

# reshape the data to match the input shape of the model
input_data = scaled_data.reshape(1, 1, 336, 5)

# convert the input data to a PyTorch tensor and send it to the CPU
input_tensor = torch.Tensor(input_data).cpu()

# set seed for reproducibility
seed = 88
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Define CNN model
class Alex(nn.Module):
  def __init__(self, output_size):
    super(Alex, self).__init__()
    self.output = output_size

    self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7,stride=1,padding=3),
                                nn.ReLU(True),)
    
    self.layer2 = nn.MaxPool2d(kernel_size=2)

    self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7,stride=1,padding=3),
                                nn.ReLU(True),)
    
    self.layer4 = nn.MaxPool2d(kernel_size=2)

    self.layer5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7,stride=1,padding=3),
                                nn.ReLU(True),)
    
    self.layer6 = nn.Sequential(nn.Linear(128*84*1, 256),
                                nn.Linear(256,168))

  def forward(self, input):
    repre = self.layer1(input)
    repre = self.layer2(repre)
    repre = self.layer3(repre) 
    repre = self.layer4(repre)
    repre = self.layer5(repre)
    repre = self.layer6(repre.view(repre.size(0), -1))
    return repre

output_size = 24 * 7 # predict 7 days of hourly data
model = Alex(output_size)
# load the saved model from the file
model.load_state_dict(torch.load('./pickle_files/model_temp.pth', map_location=torch.device('cpu')))

# make predictions using the loaded model and the input data
with torch.no_grad():
    predictions = model(input_tensor).cpu().numpy()

# convert the predictions back to the original scale
forecast_temp = scaler_temp.inverse_transform(predictions.reshape(-1, 1))

# load the saved model from the file
model.load_state_dict(torch.load('./pickle_files/model_precip.pth', map_location=torch.device('cpu')))

# make predictions using the loaded model and the input data
with torch.no_grad():
    predictions = model(input_tensor).cpu().numpy()

# convert the predictions back to the original scale
forecast_precip = scaler_precip.inverse_transform(predictions.reshape(-1, 1))

# create a datetime index starting from 2023-01-01 00:00:00 with hourly frequency
index = pd.date_range(start='2023-01-01 00:00:00', periods=168, freq='H')

# create a dataframe with the forecast values and the datetime index
df_forecast = pd.DataFrame({'Temperature': forecast_temp.flatten(), 'Precipitation': forecast_precip.flatten()}, index=index)
df_forecast['Precipitation'] = df_forecast['Precipitation'].clip(lower=0)

# plot the forecast values
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# plot forecast_temp in the first subplot
axs[0].plot(df_forecast.index, df_forecast['Temperature'], color='blue')
axs[0].set_title('Temperature Forecast')

# plot forecast_precip in the second subplot
axs[1].plot(df_forecast.index, df_forecast['Precipitation'], color='green')
axs[1].set_title('Precipitation Forecast')

# set shared x-axis label and adjust spacing between subplots
fig.tight_layout(pad=3.0)
plt.xlabel('Date')
plt.show()

df_forecast.to_csv('df_forecast.csv')
