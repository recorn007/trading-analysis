import pandas as pd
import numpy as np
import matplotlib.finance as finplt
import matplotlib.pyplot as plt
import glob
import datetime as dt
from MyModules.features import new_datetime
from matplotlib.patches import Rectangle

df = pd.read_csv(r'dukascopy - EURUSD_Candlestick_4_Hour_BID_31.12.2015-30.12.2016.csv',
                 parse_dates=[0], index_col=0, date_parser=lambda d: pd.datetime.strptime(d[:13], '%d.%m.%Y %H'))
df = df[(df.Open != df.High) & (df.Open != df.Low) & (df.Open != df.Close)]
pd.set_option("display.max_columns", 999)

# Define the window into the df, for a specified range of time. Also take into account the longterm trend behind the window
min_w_lt = 100
min_w = 550
max_w = 650

df_longterm = df.iloc[min_w_lt:max_w, :].copy()  # long term context
df_window = df.iloc[min_w:max_w, :].copy()       # short term window

cols = ['Candle Pattern', 'Same-sized Candle Trend Rejection', 'Engulfing Pattern', 'Immediate Trend Direction', 'Rejection',
        'Near Short-term Control', 'Near Long-term Control', 'In Excess Above Short-term Value', 'In Excess Below Short-term Value',
        'In Excess Above Long-term Value', 'In Excess Below Long-term Value', 'Rejected Short-term Control', 'Rejected Long-term Control',
        'Rejected Short-term Upper Limit', 'Rejected Short-term Lower Limit', 'Rejected Long-term Upper Limit', 'Rejected Long-term Lower Limit',
        'Near Short-term SR', 'Near Long-term SR', 'Near Sloped SR', 'Rejected Short-term SR line', 'Rejected Long-term SR line',
        'Rejected Sloped SR line', 'Long-term Trend Direction', 'In Excess of Long-term Value Area, Trend-following',
        'In Excess of Long-term Value Area, Counter-trend', 'Rejected Fibo level 236', 'Rejected Fibo level 382', 'Rejected Fibo level 618']
for c in cols:
    df_window[c] = np.array(np.nan)

def plot_ticks(df_window):
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = (20, 15)
    fig.tight_layout()
    plt.gca().get_xaxis().set_visible(False)
    #fig.subplots_adjust(bottom=0.2)
    finplt.candlestick2_ohlc(ax, df_window.Open, df_window.High, df_window.Low, df_window.Close,
                             width=0.6, colorup='g', colordown='r', alpha=0.75)
    #date_labels = df_window.index.map(lambda d: str(d)[-14:][:5] + ' | ' + str(d)[-8:][:2] + 'h')
    #plt.xticks(range(len(df_window)), date_labels)
    ax.autoscale_view()
    #plt.setp(plt.gca().get_xticklabels(), color='lightgrey', fontsize=8, rotation=45, horizontalalignment='right')
    #plt.gca().get_xticklabels()[100].set_color('red')
    ax.add_patch(Rectangle((99, 0), 2, 10, facecolor='purple', alpha=0.15))    
    plt.show()

iteration = 0
plot_ticks(df_window.append(df.iloc[max_w+iteration : max_w+iteration+100, :]))

toTrade = []
print(df_window.index[-1])
print(df.index[max_w])

while True:
    df_longterm, df_window, shortterm_SR, longterm_SR, shortterm_trend, st_lower, st_upper, longterm_trend, lt_lower, lt_upper, sloped_sr_lines, sloped_sr_lines_starts \
        = new_datetime(df_longterm, df_window, df.iloc[max_w+iteration, :], pip_closeness_tol=0.0008)
    show = df_window.append(df.iloc[max_w+iteration+1 : max_w+iteration+101, :])
    plot_ticks(show)
    
    trade = input('What to do at ' + str(show.index[100])[:13] + 'h:  ')
    if trade == 'undo':
        iteration -= 2
        toTrade = toTrade[:-1]
    elif trade == 'done':
        break
    else:
        toTrade = np.append(toTrade, trade)
    iteration += 1

toTrade = pd.DataFrame(toTrade, index=df.index[max_w : max_w+iteration], columns=['Category'])

filename = 'trades.csv' if not glob.glob(filename) else 'trades (1).csv'
toTrade.to_csv(filename)