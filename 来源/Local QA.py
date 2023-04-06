from datetime import datetime
from datetime import timedelta
import pandas as pd


def get_hous(s):
    if not 0 < int(s[:-2]) < 24:
        print('hous must in [0,24]')
        return False
    if 'am' in s:
        if int(s[:-2]) > 12:
            print('am hous must in [0,12]')
            return False

        return s[:-2]

    elif 'pm' in s:
        h = s[:-2]
        if int(h) < 12:
            return str(int(h) + 12)
        return h


def weather(text):
    if ('am' not in text) and ('pm' not in text):
        # print('palce select am or pm')
        while True:
            x = input('place input am or pm').lower()
            if x == 'am' or x == 'pm':
                text = text+x
                break
            print('only am or pm,retry!')

    key1 = {
        'temperature': 0,
        'temp': 0,
        'precipitation': 1
    }
    datetime(2017, 10, 7)
    now_time = '2022/12/31'
    now_time_format = [int(i) for i in now_time.split('/')]
    now_datetime = datetime(now_time_format[0], now_time_format[1], now_time_format[2])
    print('now date:', now_datetime.strftime('%Y-%m-%d'))

    clas = max([v if k in text else -999 for k, v in key1.items()])

    key2 = {
        'add options': '',
        'yesterday': 1,
        'today': 0,
        'tomorrow': -1,
        'tmr': -1,

        '2023.01.01': -1,
        '2023.01.02': -2,
        '2023.01.03': -3,
        '2023.01.04': -4,
        '2023.01.05': -5,
        '2023.01.06': -6,
        '2023.01.07': -7,

        '2023/01/01': -1,
        '2023/01/02': -2,
        '2023/01/03': -3,
        '2023/01/04': -4,
        '2023/01/05': -5,
        '2023/01/06': -6,
        '2023/01/07': -7,

        '01/01/2023': -1,
        '01/02/2023': -2,
        '01/03/2023': -3,
        '01/04/2023': -4,
        '01/05/2023': -5,
        '01/06/2023': -6,

        '01/07/2023': -7,
        '01.01.2023': -1,
        '02.01.2023': -2,
        '03.01.2023': -3,
        '04.01.2023': -4,
        '05.01.2023': -5,
        '06.01.2023': -6,
        '07.01.2023': -7,

        '01.01.2023': -1,
        '02.01.2023': -2,
        '03.01.2023': -3,
        '04.01.2023': -4,
        '05.01.2023': -5,
        '06.01.2023': -6,
        '07.01.2023': -7,

        'jan1': -1,
        'jan2': -2,
        'jan3': -3,
        'jan4': -4,
        'jan5': -5,
        'jan6': -6,
        'jan7': -7,

        'Jan1': -1,
        'Jan2': -2,
        'Jan3': -3,
        'Jan4': -4,
        'Jan5': -5,
        'Jan6': -6,
        'Jan7': -7,

        'Jan01': -1,
        'Jan02': -2,
        'Jan03': -3,
        'Jan04': -4,
        'Jan05': -5,
        'Jan06': -6,
        'Jan07': -7,

        'jan.01': -1,
        'jan.02': -2,
        'jan.03': -3,
        'jan.04': -4,
        'jan.05': -5,
        'jan.06': -6,
        'jan.07': -7,

        'Jan.01': -1,
        'Jan.02': -2,
        'Jan.03': -3,
        'Jan.04': -4,
        'Jan.05': -5,
        'Jan.06': -6,
        'Jan.07': -7,

        'january1 ': -1,
        'january2': -2,
        'january3': -3,
        'january4': -4,
        'january5': -5,
        'january6': -6,
        'january7': -7,

        'January1': -1,
        'January2': -2,
        'January3': -3,
        'January4': -4,
        'January5': -5,
        'January6': -6,
        'January7': -7,

        'January01': -1,
        'January02': -2,
        'January03': -3,
        'January04': -4,
        'January05': -5,
        'January06': -6,
        'January07': -7,

        'January.01': -1,
        'January.02': -2,
        'January.03': -3,
        'January.04': -4,
        'January.05': -5,
        'January.06': -6,
        'January.07': -7,

        'first of jan': -1,
        'second of jan': -2,
        'third of jan': -3,

        'first of Jan': -1,
        'second of Jan': -2,
        'third of Jan': -3,

        'First of Jan': -1,
        'Second of Jan': -2,
        'Third of Jan': -3,

    }
    times = max([v if k in text else -999 for k, v in key2.items()])
    tar_time = (now_datetime - timedelta(days=times)).strftime('%Y/%m/%d').replace('/0', '/')

    hous_ = text.split(' ')[-1]
    assert get_hous(hous_)
    hous = get_hous(hous_)
    h = hous + ':00'.title()
    tar_date = tar_time + ' ' + h
    t_list, p_list = [], []
    data_df = pd.read_csv('df_forecast.csv')
    for index, row in data_df.iterrows():
        # 检查date列的值是否等于'2022/12/31 15:00'
        if row['date'] == tar_date:
            # 把这一行的其他三个数据赋值给a,b,c
            if clas == 0:
                result = row['Temperature']
            else:
                result = row['Precipitation']
        if row['date'].split(' ')[0] == tar_time:
            t_list.append(row['Temperature'])
            p_list.append(row['Precipitation'])
    if clas == 0:
        avg = sum(t_list) / len(t_list)
        mx = max(t_list)
        mn = min(t_list)
    else:
        avg = sum(p_list) / len(p_list)
        mx = max(p_list)
        mn = min(p_list)
    c = {v: k for k, v in key1.items()}[clas]
    print({v: k for k, v in key2.items()}[times] + ' {} '.format(h) + c + ' is ' + str(result) + ', average :'+c + ':'+ str(avg))
    print(f'max {c}:{mx},min {c}:{mn}')

if __name__ == '__main__':
    Q = '''What's the Temperature like 2023.01.04 4pm'''.lower()
    #weather(Q)
    while True:
        Q = input('Please input the question').lower()
        try:
            weather(Q)

        except Exception:
            print('re input')
