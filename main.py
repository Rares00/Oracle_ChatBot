import csv
import requests
import pandas as pd

CSV_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point?Time=LST&parameters=T2M,RH2M,PRECTOTCORR,PS,WS50M&community=RE&longitude=-0.1677&latitude=51.4627&start=20180101&end=20230101&format=CSV"
CSV_FILE = "weather_data.csv"

try:
    response = requests.get(CSV_URL)

    # Check if the request was successful
    if response.status_code == 200:
        with requests.Session() as s:
            download = s.get(CSV_URL)

            decoded_content = download.content.decode('utf-8')

            csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')
            data = list(csv_reader)
    else:
        raise Exception("API request failed with status code {}".format(response.status_code))
except:
    # If the API request failed, try reading data from the CSV file
    with open(CSV_FILE, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

# Use the data obtained from either the API or the CSV file


data = data[13:]



file = open('weather_data_updated.csv', 'w', newline ='')
 
# writing the data into the file
with file:   
    write = csv.writer(file)
    write.writerows(data)

df = pd.read_csv("weather_data_updated.csv")

print(df)
