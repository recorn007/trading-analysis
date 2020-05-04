import numpy as np, datetime as dt
import mpl_finance as finplt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from IPython.display import clear_output

# modify parameters as needed

cat_map = {0: "No category", 1: "Hammer (body near high)", 2: "Inverted hammer (body near low)", 3: "Spinning top", 4: "Doji with close near high", 5: "Doji with close near low", 6: "Doji with close near middle", 7: "Marubozu", 8: "Hanging man", 9: "Shooting star"}
params = {0: {'color': 'c', 'linewidth': 2.35, 'alpha': 1}, 1: {'color': 'c', 'linewidth': 2, 'alpha': 0.55}, 2: {'color': 'c', 'linewidth': 2, 'alpha': 0.55}, 3: {'color': 'green', 'linewidth': 2.35, 'alpha': 1}, 4: {'color': 'green', 'linewidth': 2, 'alpha': 0.4}, 5: {'color': 'green', 'linewidth': 2, 'alpha': 0.4}}

def plot_ticks(df_window, longterm_SR, shortterm_SR, longterm_trend, lt_lower, lt_upper, shortterm_trend, st_lower, st_upper, sloped_sr_lines, sloped_sr_lines_starts, last_date, len_of_future_bars, instr, period, len_longterm, len_window, txtOutput=True):
    clear_output()
    plt.rcParams['figure.figsize'] = (16, 8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    finplt.candlestick2_ohlc(ax, df_window.Open, df_window.High, df_window.Low, df_window.Close,
                             width=0.6, colorup='lime', colordown='r', alpha=0.75)    
    ax.autoscale_view()
    ax.set_facecolor('#161616')
# Highlight last closed candle
    ax.add_patch(Rectangle((len(df_window)-len_of_future_bars-0.5, 0), 1.2, max(df_window.High), facecolor='purple', alpha=0.32))
# Plot the SR lines
    for line in (lt for lt in longterm_SR if lt >= min(df_window.Low) and lt <= max(df_window.High)):
        plt.axhline(y=line, color='firebrick', linewidth=1.5)
    for line in shortterm_SR:
        plt.axhline(y=line, color='royalblue', linewidth=0.8)
# Plot the trend lines
    for i, trend in enumerate((shortterm_trend, st_lower, st_upper, longterm_trend, lt_lower, lt_upper)):
        lm = LinearRegression(n_jobs=-1)
        try:
            lm.fit(np.arange(5).reshape(-1, 1), trend[:5])
        except:
            print(i, trend)
            exit
        y = lm.predict(np.arange(len(df_window)).reshape(-1, 1))
        plt.plot(y, '-.', color=params[i]['color'], linewidth=params[i]['linewidth'], alpha=params[i]['alpha'])
# Plot S+R lines with same slope as trend's control, determined by trend rejection candles
    for i in range(len(sloped_sr_lines)):
        if sloped_sr_lines[i] != []:
            lm = LinearRegression(n_jobs=-1)
            lm.fit(np.arange(len(sloped_sr_lines[i])).reshape(-1, 1), sloped_sr_lines[i])
            y = lm.predict(np.arange(len(df_window)-df_window.index.get_loc(sloped_sr_lines_starts[i])).reshape(-1, 1))
            plt.plot(np.add(df_window.index.get_loc(sloped_sr_lines_starts[i]), range(len(df_window)-df_window.index.get_loc(sloped_sr_lines_starts[i]))),
                     y, '--', color='purple', linewidth=2, alpha=0.7)
# Format axes
    plt.ylim(min(df_window.Low), max(df_window.High))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
# Format ticks
    if period == 'M':
        maj_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-2)]) if (int(str(d)[5:7])<=2 and i%2 == 0) or d == last_date]
        maj_date_labels = df_window.index.map(lambda d: str(d)[2:10])
        min_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-2)]) if i not in maj_date_ticks and i%2 == 0]
        min_date_labels = df_window.index.map(lambda d: str(d)[5:10] + ' ')
    elif period == 'W':
        maj_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-3)]) if (str(d)[5:7]=='01' and str(df_window.index[i-1])[5:7]=='12' and str(df_window.index[i-1])[5:7]!='01') or d == last_date]
        maj_date_labels = df_window.index.map(lambda d: str(d)[:4] + '          ')
        min_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-3)]) if i not in maj_date_ticks and i%2 == 0]
        min_date_labels = df_window.index.map(lambda d: str(d)[5:10])
    elif period == 'D':
        maj_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-3)]) if (str(d)[5:7]!=str(df_window.index[i-1])[5:7] and str(d)[5:7]==str(df_window.index[i+1])[5:7]) or d == last_date]
        maj_date_labels = df_window.index.map(lambda d: str(d)[5:10] + '         ')
        min_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-3)]) if i not in maj_date_ticks and i%2 == 0]
        min_date_labels = df_window.index.map(lambda d: str(d)[5:10])
    elif period == 'H4' or period == 'H1':
        maj_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-3)]) if str(d)[11:13]=='00' and i%2 == 0]
        maj_date_labels = df_window.index.map(lambda d: str(d)[5:10] + '     ')
        min_date_ticks = [i for i, d in enumerate(df_window.index[:-(len_of_future_bars-3)]) if (i not in maj_date_ticks and i%2 == 0) or d == last_date]
        min_date_labels = df_window.index.map(lambda d: str(d)[5:13])

    ax.set_xticks(min_date_ticks, minor=True)
    ax.set_xticklabels(min_date_labels[min_date_ticks], minor=True)
    ax.set_xticks(maj_date_ticks)
    ax.set_xticklabels(maj_date_labels[maj_date_ticks])

    plt.setp(plt.gca().get_xticklabels(minor=True), horizontalalignment='right', rotation=80)
    plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right', rotation=80)
    ax.tick_params(axis='x', labelsize=10)

# Save plot as jpg
    filename = './Analyses/{}/{} - {} {} in {} {}'.format(instr, ('M', 'W', 'D', 'H4', 'H1').index(period) + 1, instr, period, len_longterm, len_window)
    plt.savefig(filename + '.jpg', bbox_inches='tight')
    plt.cla() 
    plt.clf() 
    plt.close('all')

# Save features as txt
    if txtOutput:
        with open(filename + '.txt', 'w') as txt:
            print('   ' + str(df_window.index[-len_of_future_bars])[:13] + 'h', file=txt)
            txt.close()
        with open(filename + '.txt', 'a') as txt:
            if df_window.iloc[-len_of_future_bars, 5] != 0:
                print(cat_map[df_window.iloc[-len_of_future_bars, 5]], file=txt)
            for i in range(6, 34):
                if i not in (9, 28) and df_window.iloc[-len_of_future_bars, i] != 0:
                    print(df_window.columns[i], end='', file=txt)
                    if i == 8: print(':', 'Upwards' if df_window.iloc[-len_of_future_bars, i] == 1 else 'Downwards', end='', file=txt)
                    if i in (16, 17, 18, 19, 20, 21, 25, 26, 31, 32, 33): print(' (from BELOW)' if df_window.iloc[-len_of_future_bars, i] == 1 else '(from ABOVE)', file=txt)
                    else: print('', file=txt)
            txt.close()