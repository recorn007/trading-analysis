# THIS IS FOR PLOTTING WITHIN JUPYTER NOTEBOOK
# MAY NEED TO MODIFY IF PLOTTING OUTSIDE OF JUPYTER NOTEBOOK

import numpy as np
import matplotlib.finance as finplt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from IPython.display import clear_output

# modify parameters as needed

len_of_future_bars = 50
cat_map = {0: "No category", 1: "Hammer (body near high)", 2: "Inverted hammer (body near low)", 3: "Spinning top", 4: "Doji with close near high", 5: "Doji with close near low", 6: "Doji with close near middle", 7: "Marubozu", 8: "Hanging man", 9: "Shooting star"}
params = {0: {'color': 'teal', 'linewidth': 3, 'alpha': 1}, 1: {'color': 'teal', 'linewidth': 2, 'alpha': 0.6}, 2: {'color': 'teal', 'linewidth': 2, 'alpha': 0.6}, 3: {'color': 'green', 'linewidth': 3, 'alpha': 1}, 4: {'color': 'green', 'linewidth': 2, 'alpha': 0.6}, 5: {'color': 'green', 'linewidth': 2, 'alpha': 0.6}}

def plot_ticks(df_window, longterm_SR, shortterm_SR, longterm_trend, lt_lower, lt_upper, shortterm_trend, st_lower, st_upper, sloped_sr_lines, sloped_sr_lines_starts):
    clear_output()
    plt.rcParams['figure.figsize'] = (24, 14)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    finplt.candlestick2_ohlc(ax, df_window.Open, df_window.High, df_window.Low, df_window.Close,
                             width=0.6, colorup='g', colordown='r', alpha=0.75)    
    ax.autoscale_view()
    ax.add_patch(Rectangle((len(df_window)-len_of_future_bars-0.5, 0), 1.2, 10, facecolor='purple', alpha=0.3))
# Plot the SR lines
    for line in (lt for lt in longterm_SR if lt >= min(df_window.Low) and lt <= max(df_window.High)):
        plt.axhline(y=line, color='r', linewidth=2)
    for line in shortterm_SR:
        plt.axhline(y=line, color='b', linewidth=0.8)
# Plot the trend lines
    for i, trend in enumerate((shortterm_trend, st_lower, st_upper, longterm_trend, lt_lower, lt_upper)):
        lm = LinearRegression(n_jobs=-1)
        lm.fit(np.arange(5).reshape(-1, 1), trend[:5])
        y = lm.predict(np.arange(len(df_window)).reshape(-1, 1))
        plt.plot(y, '-.', color=params[i]['color'], linewidth=params[i]['linewidth'], alpha=params[i]['alpha'])
# Plot S+R lines with same slope as trend's control, determined by trend rejection candles
    for i in range(len(sloped_sr_lines)):
        if sloped_sr_lines[i] != []:
            lm = LinearRegression(n_jobs=-1)
            lm.fit(np.arange(len(sloped_sr_lines[i])).reshape(-1, 1), sloped_sr_lines[i])
            y = lm.predict(np.arange(len(df_window)-df_window.index.get_loc(sloped_sr_lines_starts[i])).reshape(-1, 1))
            plt.plot(np.add(df_window.index.get_loc(sloped_sr_lines_starts[i]), range(len(df_window)-df_window.index.get_loc(sloped_sr_lines_starts[i]))),
                     y, '--', color='purple', linewidth=3, alpha=0.7)
# Format axes
    plt.ylim(min(df_window.Low), max(df_window.High))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    [i.set_color("white") for i in plt.gca().get_yticklabels()]
    
    maj_date_ticks = [i for i, d in enumerate(df_window.index) if i%2 == 0 and i != len(df_window)-1 and str(d)[:10] != str(df_window.index[i+1])[:10]]
    maj_date_labels = df_window.index.map(lambda d: str(d)[5:10] + '  ' + str(d)[2:4])
    ax.set_xticks(maj_date_ticks)
    ax.set_xticklabels(maj_date_labels[maj_date_ticks])
    plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right', rotation=80)
    [i.set_color("white") for i in plt.gca().get_xticklabels()]
    ax.tick_params(axis='x', labelsize=10)
    
    plt.show()
    
    print('   ' + str(df_window.index[-len_of_future_bars+1])[:13] + 'h')
    if df_window.iloc[-len_of_future_bars+1, 5] != 0:
        print(cat_map[df_window.iloc[-len_of_future_bars+1, 5]])
    for i in range(6, 34):
        if i not in (9, 28) and df_window.iloc[-len_of_future_bars+1, i] != 0:
            print(df_window.columns[i], end='')
            if i == 8: print(':', 'Upwards' if df_window.iloc[-len_of_future_bars+1, i] == 1 else 'Downwards', end='')
            if i in (16, 17, 18, 19, 20, 21, 25, 26, 31, 32, 33): print(' (from BELOW)' if df_window.iloc[-len_of_future_bars+1, i] == 1 else '(from ABOVE)')
            else: print('')