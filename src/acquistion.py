import csv
import requests

def get_weather_data(CSV_URL, CSV_FILE):
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
    return data
# Use the data obtained from either the API or the CSV file