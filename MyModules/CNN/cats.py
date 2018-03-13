import pandas as pd
import numpy as np
import matplotlib.finance as finplt
import matplotlib.pyplot as plt
import glob

def plot_ticks(df_window):
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = (8, 6)
    fig.subplots_adjust(bottom=0.2)

    # Plot the candlesticks
    finplt.candlestick2_ohlc(ax, df_window.Open, df_window.High, df_window.Low, df_window.Close,
                             width=0.6, colorup='g', colordown='r', alpha=0.75)

    date_labels = df_window.index.map(lambda d: str(d)[-14:][:5] + ' | ' + str(d)[-8:][:2] + 'h')
    plt.xticks(range(len(df_window)), date_labels)
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), color='lightgrey', fontsize=8, rotation=45,   
             horizontalalignment='right')

    plt.gca().get_xticklabels()[10].set_color('red')

    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((9, 0), 2, 10, facecolor='purple', alpha=0.15))

    u = df_window.Open[10] if df_window.Open[10] >= df_window.Close[10] else df_window.Close[10]
    l = df_window.Open[10] if df_window.Open[10] <= df_window.Close[10] else df_window.Close[10]

    ax.annotate('upper tail:   ' + str(round((df_window.High[10]-u) / (u-l), 2)), xy=(0.42, 0.94), xycoords='axes fraction', fontsize=9)
    ax.annotate('lower tail:   ' + str(round((l-df_window.Low[10]) / (u-l), 2)), xy=(0.42, 0.88), xycoords='axes fraction', fontsize=9)

    plt.show()



df = pd.read_csv(r'/home/michael/Desktop/forex/dukascopy - EURUSD_Candlestick_4_Hour_BID_31.12.2015-30.12.2016.csv', parse_dates=[0], index_col=0, date_parser=lambda d: pd.datetime.strptime(d[:13], '%d.%m.%Y %H'))

iteration = 0
cats = []

while True:
    show = df.iloc[1115+iteration:1145+iteration, :]      # continuing iteration from first run. originally first number was 49+iteration
    if (show.Open[10] == show.High[10]) & (show.Open[10] == show.Low[10]) & (show.Open[10] == show.Close[10]):
        cats = np.append(cats, 'delete this row!')
    else:
        plot_ticks(show)
        cat = input('Pattern for ' + str(show.index[10])[:13] + 'h:  ')
        if cat == 'undo':
            iteration -= 2
            cats = cats[:-1]
        elif cat == 'done':
            break
        else:
            cats = np.append(cats, cat)
    iteration += 1

cats = pd.DataFrame(cats, index=df.index[1125:1125+iteration], columns=['Category'])
cats = cats[cats['Category'] != 'delete this row!']

filename = 'y candles.csv'
files_present = glob.glob(filename)
if not files_present:
    cats.to_csv(filename)
else:
    cats.to_csv('y candles (1).csv')