from datetime import datetime
from datetime import timedelta
import pandas as pd

##读取当前时间
def get_date(days):
    now_date = datetime.now()
    return (now_date - timedelta(days=days+180)).strftime('%Y-%m-%d')
##加180是没有当前数据


def weather(text):
    key = {
        'add options': '',
        'yesterday': 1,
        'today': 0,
        'tomorrow': -1

    }
    # max([v if k in text else -999 for k,v in enumerate(key)])
    cl1 = max([v if k in text else -999 for k, v in key.items()])
    date = get_date(cl1)

    Q = {
        'weather': 1,
        'temperature': 0,
        'humidity': -1,
        'precipitation': -2,
        'wind speed': -3,

    }
    headers = {
        'weather': '',
        'temperature': 'T2M',
        'humidity': 'RH2M',
        'precipitation': 'PRECTOTCORR',
        'wind speed': 'WS50M',

    }
    cl2 = max([v if k in text else -999 for k, v in Q.items()])
    header = headers[{v: k for k, v in Q.items()}[cl2]]

    '''
    Replace with a crawler, and maintain the dataframe
    '''
    data_df = pd.read_csv('weather_data_updated.csv')

    dct = {
        'YEAR': str(int(date[:4])),
        'MO': str(int(date[5:7])),
        'DY': str(int(date[-2:])),
    }

    qry = ' and '.join(['{} == {}'.format(k, v) for k, v in dct.items()])
    result_ = data_df.query(qry)
    result = result_.iloc[0].loc[header]


    A = {v: k for k, v in key.items()}[cl1] + '   ' + {v: k for k, v in Q.items()}[cl2] + '   ' + str(result)

    print(A)


Q = '''What's the Temperature like '''
while True:
    Q=input().lower()
    try:
        weather(Q)
    except Exception:
        print('re-input')

