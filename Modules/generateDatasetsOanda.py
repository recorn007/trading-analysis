# import os
# if os.path.dirname(os.path.realpath(__file__))[-12:] != 'the-forex-ai':
#    print('This script can only be ran at the root folder of the-forex-ai. Please drop this file there first and then run.')
#    exit()

import pandas as pd, numpy as np, datetime as dt
import http, json, os
from MyModules.features import new_datetime_alpha

def main():
    currensy = ('EUR_USD', 'EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'AUD_USD', 'USD_CAD', 'USD_JPY')
    timePeriods = ('M', 'W', 'D', 'H4', 'H1')

    check_dir(currensy)

    for cur in currensy:
        for time in timePeriods:
            print("Generating the {} {} dataset".format(cur, time))
            gen_csv(cur, time)

    print('')
    
    return

def get_oanda_candles(instr, period):
    conn = http.client.HTTPSConnection("api-fxpractice.oanda.com")
    headers = {"Authorization": "Bearer ***REMOVED***"}
    url = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles?&count=550&granularity={}&alignmentTimezone=America/Los_Angeles"\
            .format(instr, period)
    conn.request("GET", url, None, headers)
    resp = conn.getresponse()
    if resp.status in (201, 200):
        return json.loads(resp.read())
    else:
        raise http.client.HTTPException("Error in HTTP request (status: " + str(resp.status) + "):\n" + str(resp.read()))

def gen_csv(c, t):
    df = get_oanda_candles(c, t)
    df = pd.DataFrame(df['candles']).drop(['complete', 'volume'], axis=1)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    cols = ('Open', 'High', 'Low', 'Close')
    for col in cols:
        df[col] = df['mid'].apply(lambda r: float(r['{}'.format(col[0].lower())]))
    df = df.drop('mid', axis=1)

    df['Rejection'] = np.nan
    for n in range(4, 9): df[str(n)] = np.nan
    df = df[['Open', 'High', 'Low', 'Close', '4', '5', '6', '7', '8', 'Rejection']]

    len_window = 120
    min_w = max(0, len(df)-len_window)

    df_get_alpha = df.iloc[min_w:, :]
    df = df.iloc[:min_w, :]

    for i in range(len(df_get_alpha)):
        _, df = new_datetime_alpha(df, df, df_get_alpha.iloc[i])

    df = df.drop([str(n) for n in range(4, 9)], axis=1)
    df.to_csv(r'./Datasets/{} {}.csv'.format(c, t))

    return

def check_dir(currensy):
    if not os.path.exists(r'./Datasets'): os.mkdir(r'./Datasets')
    if not os.path.exists(r'./Analyses'): os.mkdir(r'./Analyses')

    for cur in currensy:
        if not os.path.exists(r'./Analyses/{}'.format(cur)):
            os.mkdir(r'./Analyses/{}'.format(cur))

if __name__ == '__main__':
    main()