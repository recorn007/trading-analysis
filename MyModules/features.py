import pandas as pd
import numpy as np
import mpl_finance as finplt
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.linear_model import LinearRegression
from PIL import Image
from keras.models import load_model

CNNmodel = load_model('./MyModules/An eye for an eye - a CNN model.h5')

def git_candle_cat(to_plot, model_guess_tol=0.60):  # model_guess_tol: minimum required certainty for candle category; otherwise cat=0
    plt.rcParams['figure.figsize'] = (0.4, 0.8)
    fig, ax = plt.subplots()
    finplt.candlestick2_ohlc(ax, to_plot.Open, to_plot.High, to_plot.Low, to_plot.Close,
                         width=0.6, colorup='g', colordown='r', alpha=1)
    plt.axis('off')
    plt.savefig('./MyModules/temp_candle.jpg')
    plt.close()
    category_i = Image.open('./MyModules/temp_candle.jpg').convert('L') # convert to grey-scale
    category_i = np.asarray(category_i.getdata(), dtype=np.float64).reshape((category_i.size[1], category_i.size[0]))
    category_i = np.asarray(category_i, dtype=np.uint8) #if values still in range 0-255! 
    category_i = np.array(Image.fromarray(category_i, mode='L')).astype('float32')
    category_i /= 255
    #category_i = category_i.reshape(1, 28, 57, 1)
    category_i = category_i.reshape(1, 40, 80, 1)
    category_i = CNNmodel.predict(category_i)
    
    if np.max(category_i) >= model_guess_tol:
        category_i = np.argmax(category_i)
        if category_i in (1, 2):         # re-classify hammers as either Hanging Man or Shooting Star
            if to_plot.iloc[0, 4] == 1:  # depending on immediate trend
                if category_i == 1:
                    category_i = 8
                else:
                    category_i = 9
    else:
        category_i = 0
    
    return category_i

def immediate_trend(to_plot):
    immediate_trend = trend_lines(to_plot, t=25, control_only=True)
    return np.sign(immediate_trend[-1]-immediate_trend[0])

def same_sized_candle_trend_rejection(candles, trend, ratio):
    size_tol = 1 if 1.25 * ratio < 1 else 1.25 * ratio
    if candles.Close.iloc[0] - candles.Open.iloc[0] != 0 and candles.Close.iloc[1] - candles.Open.iloc[1] != 0:
        if (1/size_tol) <= (abs(candles.Close.iloc[0] - candles.Open.iloc[0]) / abs(candles.Close.iloc[1] - candles.Open.iloc[1])) <= size_tol and \
           np.sign(candles.Close.iloc[0] - candles.Open.iloc[0]) * np.sign(candles.Close.iloc[1] - candles.Open.iloc[1]) == -1 and \
           trend * np.sign(candles.Close.iloc[1] - candles.Open.iloc[1]) == -1:
            if trend == 1 and abs(candles.High.iloc[0] - candles.High.iloc[1]) <= 0.001 * ratio:
                return 1
            elif trend == -1 and abs(candles.Low.iloc[0] - candles.Low.iloc[1]) <= 0.001 * ratio:
                return 1
    return 0
    
def engulfing_check(candles):
    if candles.iloc[1, 0] > candles.iloc[0, 0] and candles.iloc[1, 1] < candles.iloc[0, 1]:
        return 1
    else:
        return 0
    
def rejection_price(candles):
    if candles.iloc[5] in (1, 2):
        return candles.iloc[2]
    elif candles.iloc[5] in (8, 9):
        return candles.iloc[1]
    elif candles.iloc[6] == 1:
        if candles.iloc[3] >= candles.iloc[0]:
            return candles.iloc[2]
        else:
            return candles.iloc[1]
    else:
        return np.nan

def support_resistance(price, quant, bin_seed):
    # sklearn
    bandwidth = estimate_bandwidth(price, quantile=quant, n_samples=None)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seed, n_jobs=-1)
    ms.fit(price)

    sr_lines = []
    for k in range(len(np.unique(ms.labels_))):
        my_members = ms.labels_ == k
        values = price.loc[my_members, 'Close']

        sr_lines.append(min(values))
        sr_lines.append(max(values))

    return sr_lines

def avg_sr_lines(sr_lines, ratio, longTerm=False):
    pip_tol = 0.0020 * ratio if longTerm else 0.0015 * ratio
    # average out lines too close to each other
    sr_lines = np.sort(sr_lines)
    while True:
        sr_lines = np.sort(sr_lines)
        to_del = False
        for i, val in enumerate(sr_lines):
            if (i > 0) & (val - sr_lines[i-1] <= pip_tol):
                to_del = True
                avg = (val + sr_lines[i-1]) / 2
                sr_lines = np.delete(sr_lines, [i, i-1])
                sr_lines = np.append(sr_lines, avg)
                break
        if to_del == False:
            break

    return sr_lines

def redundant_shortterm_sr(shortterm_SR, longterm_SR):
    to_del = []
    for i, val in enumerate(shortterm_SR):
        for j in longterm_SR:
            if abs(val-j) <= 0.0005:
                to_del.append(i)
    if len(to_del) > 0:
        shortterm_SR = np.delete(shortterm_SR, to_del)

    return shortterm_SR

def trend_lines(price, t, preserve_datetime=False, control_only=False):
    # Requires appropriate t value (where n=9, two tailed 95%)
    lm = LinearRegression(n_jobs=-1)
    lm.fit(np.arange(len(price)).reshape(-1, 1), price)
    y = lm.predict(np.arange(len(price)).reshape(-1, 1))
    
    if control_only:
        if preserve_datetime:
            y = pd.Series(y, index=price.index)
        return y
    
    y_err = price - y
    mean_x = np.mean(price)
    n = len(price)
    s_err = np.sum(np.power(y_err,2))

    # confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((np.arange(n)-mean_x), 2) /
    #             ((np.sum(np.power(np.arange(n),2))) - n*(np.power(mean_x, 2))))))
    confs = t * np.sqrt(abs((s_err/(n-2))*(1.0/n + (np.power((np.arange(n)-mean_x), 2) /
                ((np.sum(np.power(np.arange(n),2))) - n*(np.power(mean_x, 2)))))))
    
    if preserve_datetime:
        lower = pd.Series(y - abs(confs[0]), index=price.index)
        upper = pd.Series(y + abs(confs[0]), index=price.index)
        y = pd.Series(y, index=price.index)
    else:
        lower = y - abs(confs[0])
        upper = y + abs(confs[0])
    
    return y, lower, upper

def sloped_SRlines(df_window, shortterm_trend, ratio):
    rejections = df_window[~df_window['Rejection'].isnull()].loc[:, 'Rejection']
    slope = (shortterm_trend.iloc[-1]-shortterm_trend.iloc[0]) / (len(shortterm_trend))
    sloped_sr_lines, sloped_sr_lines_starts = get_sloped_SRlines(df_window.index, rejections, slope, ratio)

    return sloped_sr_lines, sloped_sr_lines_starts

def get_sloped_SRlines(df_window_index, rejections, slope, ratio):
    #tol = 0.00175 * ratio
    tol = 0.0017 * ratio
    sloped_lines = [[]]
    sloped_lines_starts = []
    for i, R in enumerate(rejections.index):
        if i <= 10:
            start = df_window_index[df_window_index >= R]
            sr_line = [slope * j + rejections.values[i] for j in range(len(start))]
            sloped_lines.append(sr_line)
            sloped_lines_starts = np.append(sloped_lines_starts, R)
    sloped_lines = np.array(sloped_lines[1:])
    
    # delete sloped SR lines too close to each other within a tolerance
    for i in range(len(sloped_lines)):
        if sloped_lines[i] != []:
            for j in range(len(sloped_lines)-1, 0, -1):
                if i != j and sloped_lines[j] != [] and \
                abs(sloped_lines[i][-1] - sloped_lines[j][-1]) <= tol:
                    sloped_lines[j] = []
                    
    return sloped_lines, sloped_lines_starts

def fibo_levels(max_price, min_price):
    maxMin_diff = max_price - min_price
    fib_level1 = max_price - 0.236 * maxMin_diff
    fib_level2 = max_price - 0.382 * maxMin_diff
    fib_level3 = max_price - 0.618 * maxMin_diff

    return fib_level1, fib_level2, fib_level3

def new_datetime_alpha(df_longterm, df_window, new_point, to_complete=False):
    df_window = df_window.append(new_point)         # When defining df_window, included all points up max_w. Here, max_w is now included
    df_longterm = df_longterm.append(new_point)
### Get range ratio of the df_window candles. The range is compared to the golden standard of the Daily EUR_USD chart, particularly captured on Sep 30, 2018
    maxi = max(df_window.Close) if max(df_window.Close) > max(df_window.Open) else max(df_window.Open)
    mini = min(df_window.Close) if min(df_window.Close) < min(df_window.Open) else min(df_window.Open)
    rangeRatio = (maxi - mini) / 0.06111
### Feature Engineerings
    df_window.iloc[-1, 8] = immediate_trend(df_window.Close.iloc[-11:-1])                                        # Immediate trend
    df_window.iloc[-1, 5] = git_candle_cat(df_window.iloc[len(df_window)-1:len(df_window), [0, 1, 2, 3, 8]])     # Candle Pattern. Pass OHLC + Immediate Trend columns
    # Same sized candle trend rejection
    df_window.iloc[-1, 6] = same_sized_candle_trend_rejection(df_window.iloc[len(df_window)-2:len(df_window), :4], df_window.iloc[-1, 8], rangeRatio)
    df_window.iloc[-1, 7] = engulfing_check(df_window.iloc[len(df_window)-2:len(df_window), 1:3])                # Engulfing pattern
    df_window.iloc[-1, 9] = rejection_price(df_window.iloc[-1, :7])                                              # Rejection

    if to_complete:
        return df_longterm, df_window, rangeRatio
    else:
        return df_longterm, df_window

def new_datetime_complete(df_longterm, df_window, new_point, keep_df_size=False):
### Return first part of new_datetime. If df size should be kept the same, drop first row after appending last
    df_longterm, df_window, rangeRatio = new_datetime_alpha(df_longterm, df_window, new_point, to_complete=True)
    if keep_df_size:
        df_window = df_window.drop(df_window.index[0])
        df_longterm = df_longterm.drop(df_longterm.index[0])
        
### Remove previous rows for engineered features, except for 'Rejection' needed for Sloped S+Rs
 ## Is this section still needed? Or redudant, if saving csv with only Rejection anyway
    #df_window.iloc[:-1, 5:9] = np.array(np.nan)
    #df_window.iloc[:-1, 10:] = np.array(np.nan)

### Regular + sloped SR lines and trends
    # Calculate the support+resistance lines
    longterm_SR = support_resistance(df_longterm.iloc[:, :4], quant=0.2, bin_seed=True)
    longterm_SR = avg_sr_lines(longterm_SR, rangeRatio, longTerm=True)
    shortterm_SR = support_resistance(df_window.iloc[:, :4], quant=0.12, bin_seed=True)
    shortterm_SR = avg_sr_lines(shortterm_SR, rangeRatio)
     # delete short-term SR lines too close to long-term SR
    shortterm_SR = redundant_shortterm_sr(shortterm_SR, longterm_SR)

    # Calculate trend lines
    shortterm_trend, st_lower, st_upper = trend_lines(df_window.Close, t=25, preserve_datetime=True)
    longterm_trend, lt_lower, lt_upper = trend_lines(df_longterm.Close, t=25, preserve_datetime=True)

    # Calculate S+R lines with same slope as trend's control, determined by trend rejection candles
    sloped_sr_lines, sloped_sr_lines_starts = sloped_SRlines(df_window, shortterm_trend, rangeRatio)

### Value area features
    pip_closeness_tol = 0.0008 * rangeRatio

    # Find whether close price is near control
    df_window.iloc[-1, 10] = 1 if abs(df_window.Close.iloc[-1]-shortterm_trend.iloc[-1]) <= pip_closeness_tol else 0
    
    # try while still getting reindex issue
    try:
        df_window.iloc[-1, 11] = 1 if abs(df_window.Close.iloc[-1]-longterm_trend.reindex(df_window.index, axis=0).iloc[-1]) <= pip_closeness_tol else 0
    except:
        print('Got the reindex error. Investigate this ish.')
        return df_longterm, df_window, shortterm_SR, longterm_SR, shortterm_trend, st_lower, st_upper, longterm_trend, lt_lower, lt_upper, sloped_sr_lines, sloped_sr_lines_starts
    
    # Find whether close price is in excess of short-term value
    df_window.iloc[-1, 12] = 1 if df_window.Close.iloc[-1] > st_upper.iloc[-1] else 0
    df_window.iloc[-1, 13] = 1 if df_window.Close.iloc[-1] < st_lower.iloc[-1] else 0
    # Find whether close price is in excess of long-term value
    df_window.iloc[-1, 14] = 1 if df_window.Close.iloc[-1] > lt_upper.reindex(df_window.index, axis=0).iloc[-1] else 0
    df_window.iloc[-1, 15] = 1 if df_window.Close.iloc[-1] < lt_lower.reindex(df_window.index, axis=0).iloc[-1] else 0
    # Find whether price rejected short-term control. Check for: positive immediate trend, high above control, close below control, close fairly far from control (for first case)
    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= shortterm_trend.iloc[-1] - pip_closeness_tol and \
    df_window.Close.iloc[-1] < shortterm_trend.iloc[-1] and \
    abs(df_window.Close.iloc[-1] - shortterm_trend.iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 16] = 1
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.Low.iloc[-1] <= shortterm_trend.iloc[-1] + pip_closeness_tol and \
    df_window.Close.iloc[-1] > shortterm_trend.iloc[-1] and \
    abs(df_window.Close.iloc[-1] - shortterm_trend.iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 16] = 2
    else:
        df_window.iloc[-1, 16] = 0
    # Find whether price rejected long-term control
    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= longterm_trend.reindex(df_window.index, axis=0).iloc[-1] - pip_closeness_tol and \
    df_window.Close.iloc[-1] < longterm_trend.reindex(df_window.index, axis=0).iloc[-1] and \
    abs(df_window.Close.iloc[-1] - longterm_trend.reindex(df_window.index, axis=0).iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 17] = 1
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.Low.iloc[-1] <= longterm_trend.reindex(df_window.index, axis=0).iloc[-1] + pip_closeness_tol and \
    df_window.Close.iloc[-1] > longterm_trend.reindex(df_window.index, axis=0).iloc[-1] and \
    abs(df_window.Close.iloc[-1] - longterm_trend.reindex(df_window.index, axis=0).iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 17] = 2
    else:
        df_window.iloc[-1, 17] = 0
    # Find whether price rejected short-term limit.
    # Positive immediate trend: high above upper limit, close below upper limit, close fairly far from limit
    # Negative immediate trend: low above lower limit, close above lower limit, close fairly far from limit
    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= st_upper.iloc[-1] - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < st_upper.iloc[-1] and \
    abs(df_window.Close.iloc[-1] - st_upper.iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 18] = 1  # rejected short-term upper limit from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.High.iloc[-1] <= st_upper.iloc[-1] + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > st_upper.iloc[-1] and \
    abs(df_window.Close.iloc[-1] - st_upper.iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 18] = 2  # rejected short-term upper limit from above
    else:
        df_window.iloc[-1, 18] = 0
    if df_window.iloc[-1, 8] == 1 and \
    df_window.Low.iloc[-1] >= st_lower.iloc[-1] - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < st_lower.iloc[-1] and \
    abs(df_window.Close.iloc[-1] - st_lower.iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 19] = 1  # rejected short-term lower limit from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.Low.iloc[-1] <= st_lower.iloc[-1] + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > st_lower.iloc[-1] and \
    abs(df_window.Close.iloc[-1] - st_lower.iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 19] = 2  # rejected short-term lower limit from above
    else:
        df_window.iloc[-1, 19] = 0
    # Find whether price rejected long-term limit
    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= lt_upper.reindex(df_window.index, axis=0).iloc[-1] - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < lt_upper.reindex(df_window.index, axis=0).iloc[-1] and \
    abs(df_window.Close.iloc[-1] - lt_upper.reindex(df_window.index, axis=0).iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 20] = 1  # rejected long-term upper limit from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.High.iloc[-1] <= lt_upper.reindex(df_window.index, axis=0).iloc[-1] + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > lt_upper.reindex(df_window.index, axis=0).iloc[-1] and \
    abs(df_window.Close.iloc[-1] - lt_upper.reindex(df_window.index, axis=0).iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 20] = 2  # rejected long-term upper limit from above
    else:
        df_window.iloc[-1, 20] = 0
    if df_window.iloc[-1, 8] == 1 and \
    df_window.Low.iloc[-1] >= lt_lower.reindex(df_window.index, axis=0).iloc[-1] - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < lt_lower.reindex(df_window.index, axis=0).iloc[-1] and \
    abs(df_window.Close.iloc[-1] - lt_lower.reindex(df_window.index, axis=0).iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 21] = 1  # rejected long-term lower limit from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.Low.iloc[-1] <= lt_lower.reindex(df_window.index, axis=0).iloc[-1] + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > lt_lower.reindex(df_window.index, axis=0).iloc[-1] and \
    abs(df_window.Close.iloc[-1] - lt_lower.reindex(df_window.index, axis=0).iloc[-1]) > pip_closeness_tol:
        df_window.iloc[-1, 21] = 2  # rejected long-term lower limit from above
    else:
        df_window.iloc[-1, 21] = 0

    df_window.iloc[-1, 22] = 0
    df_window.iloc[-1, 23] = 0
    df_window.iloc[-1, 24] = 0
    for st in shortterm_SR:  # Check for closeness to any short-term SR lines
        if abs(st - df_window.Close.iloc[-1]) <= pip_closeness_tol/1.5:
            df_window.iloc[-1, 22] = 1
            break
    for lt in (l for l in longterm_SR if l >= min(df_window.Low) and l <= max(df_window.High)):  # Check for closeness to any long-term SR lines
        if abs(lt - df_window.Close.iloc[-1]) <= pip_closeness_tol:
            df_window.iloc[-1, 23] = 1
            break
    for i in range(len(sloped_sr_lines)):  # Check for closeness to any sloped SR lines. Different code for new datetime points?
        if sloped_sr_lines[i] != []:
            line = pd.Series(sloped_sr_lines[i], index=np.add(df_window.index.get_loc(sloped_sr_lines_starts[i]), range(len(sloped_sr_lines[i]))))
            if abs(line.iloc[-1] - df_window.Close.iloc[-1]) <= pip_closeness_tol:
                df_window.iloc[-1, 24] = 1
                break
                
    df_window.iloc[-1, 25] = 0
    df_window.iloc[-1, 26] = 0
    df_window.iloc[-1, 27] = 0
    # Check if price rejected any short-term SR lines
    for st in shortterm_SR:
        if df_window.iloc[-1, 8] == 1 and \
        df_window.High.iloc[-1] >= st - 2*pip_closeness_tol and \
        df_window.Close.iloc[-1] < st and \
        abs(df_window.Close.iloc[-1] - st) > pip_closeness_tol:
            df_window.iloc[-1, 25] += 1
        elif df_window.iloc[-1, 8] == -1 and \
        df_window.Low.iloc[-1] <= st + 2*pip_closeness_tol and \
        df_window.Close.iloc[-1] > st and \
        abs(df_window.Close.iloc[-1] - st) > pip_closeness_tol:
            df_window.iloc[-1, 25] += 1
    # Check if price rejected any long-term SR lines
    for lt in (l for l in longterm_SR if l >= min(df_window.Low) and l <= max(df_window.High)):
        if df_window.iloc[-1, 8] == 1 and \
        df_window.High.iloc[-1] >= lt - 2*pip_closeness_tol and \
        df_window.Close.iloc[-1] < lt and \
        abs(df_window.Close.iloc[-1] - lt) > pip_closeness_tol:
            df_window.iloc[-1, 26] += 1
        elif df_window.iloc[-1, 8] == -1 and \
        df_window.Low.iloc[-1] <= lt + 2*pip_closeness_tol and \
        df_window.Close.iloc[-1] > lt and \
        abs(df_window.Close.iloc[-1] - lt) > pip_closeness_tol:
            df_window.iloc[-1, 26] += 1
    # Check if price rejected any sloped SR lines
    for i in range(len(sloped_sr_lines)):
        if sloped_sr_lines[i] != []:
            line = pd.Series(sloped_sr_lines[i], index=np.add(df_window.index.get_loc(sloped_sr_lines_starts[i]), range(len(sloped_sr_lines[i]))))
            if df_window.iloc[-1, 8] == 1 and \
            df_window.High.iloc[-1] >= line.iloc[-1] - pip_closeness_tol/1.5 and \
            df_window.Close.iloc[-1] < line.iloc[-1] and \
            abs(df_window.Close.iloc[-1] - line.iloc[-1]) > pip_closeness_tol:
                df_window.iloc[-1, 27] += 1
            elif df_window.iloc[-1, 8] == -1 and \
            df_window.Low.iloc[-1] <= line.iloc[-1] + pip_closeness_tol/1.5 and \
            df_window.Close.iloc[-1] > line.iloc[-1] and \
            abs(df_window.Close.iloc[-1] - line.iloc[-1]) > pip_closeness_tol:
                df_window.iloc[-1, 27] += 1
### Long-term trend-following and counter-trend opportunities
    if abs(np.rad2deg(np.arctan2(1000*(longterm_trend[-1] - longterm_trend[0]), len(longterm_trend)))) < 8:
        df_window.iloc[-1, 28] = 0   # when slope of LTR is sideways, or under 8°
    else:
        df_window.iloc[-1, 28] = np.array(np.sign(longterm_trend[-1] - longterm_trend[0]))

    # In Excess of Long-term Value Area, Trend-following
    if df_window.iloc[-1, 28] == 1 and df_window.Close.iloc[-1] <= lt_lower.reindex(df_window.index, axis=0).iloc[-1] + pip_closeness_tol:
        df_window.iloc[-1, 29] = 1
    elif df_window.iloc[-1, 28] == -1 and df_window.Close.iloc[-1] >= lt_upper.reindex(df_window.index, axis=0).iloc[-1] - pip_closeness_tol:
        df_window.iloc[-1, 29] = 1
    else:
        df_window.iloc[-1, 29] = 0
    #In Excess of Long-term Value Area, Counter-trend
    if df_window.iloc[-1, 28] == 1 and df_window.Close.iloc[-1] >= lt_upper.reindex(df_window.index, axis=0).iloc[-1] - pip_closeness_tol:
        df_window.iloc[-1, 30] = 1
    elif df_window.iloc[-1, 28] == -1 and df_window.Close.iloc[-1] <= lt_lower.reindex(df_window.index, axis=0).iloc[-1] + pip_closeness_tol:
        df_window.iloc[-1, 30] = 1
    elif df_window.iloc[-1, 28] == 0 and \
    ((df_window.Close.iloc[-1] <= lt_lower.reindex(df_window.index, axis=0).iloc[-1] + pip_closeness_tol) or \
    (df_window.Close.iloc[-1] >= lt_upper.reindex(df_window.index, axis=0).iloc[-1] - pip_closeness_tol)):
        df_window.iloc[-1, 30] = 1
    else:
        df_window.iloc[-1, 30] = 0
### Fibo rejection checks
    fib_level1, fib_level2, fib_level3 = fibo_levels(df_window.Close.max(), df_window.Close.min())
    
    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= fib_level1 - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < fib_level1 and \
    abs(df_window.Close.iloc[-1] - fib_level1) > pip_closeness_tol:
        df_window.iloc[-1, 31] = 1  # rejected Fibo level 236 from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.High.iloc[-1] <= fib_level1 + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > fib_level1 and \
    abs(df_window.Close.iloc[-1] - fib_level1) > pip_closeness_tol:
        df_window.iloc[-1, 31] = 2  # rejected Fibo level 236 from above
    else:
        df_window.iloc[-1, 31] = 0

    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= fib_level2 - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < fib_level2 and \
    abs(df_window.Close.iloc[-1] - fib_level2) > pip_closeness_tol:
        df_window.iloc[-1, 32] = 1  # rejected Fibo level 382 from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.High.iloc[-1] <= fib_level2 + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > fib_level2 and \
    abs(df_window.Close.iloc[-1] - fib_level2) > pip_closeness_tol:
        df_window.iloc[-1, 32] = 2  # rejected Fibo level 382 from above
    else:
        df_window.iloc[-1, 32] = 0

    if df_window.iloc[-1, 8] == 1 and \
    df_window.High.iloc[-1] >= fib_level3 - 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] < fib_level3 and \
    abs(df_window.Close.iloc[-1] - fib_level3) > pip_closeness_tol:
        df_window.iloc[-1, 33] = 1  # rejected Fibo level 618 from below
    elif df_window.iloc[-1, 8] == -1 and \
    df_window.High.iloc[-1] <= fib_level3 + 2*pip_closeness_tol and \
    df_window.Close.iloc[-1] > fib_level3 and \
    abs(df_window.Close.iloc[-1] - fib_level3) > pip_closeness_tol:
        df_window.iloc[-1, 33] = 2  # rejected Fibo level 618 from above
    else:
        df_window.iloc[-1, 33] = 0
### Close above / below (fundamental)
    df_window.iloc[-1, 34] = 1 if df_window.iloc[-1, 3] > df_window.iloc[-2, 3] else 0
    df_window.iloc[-1, 35] = 1 if df_window.iloc[-1, 3] < df_window.iloc[-2, 3] else 0
    
    return df_longterm, df_window, shortterm_SR, longterm_SR, shortterm_trend, st_lower, st_upper, longterm_trend, lt_lower, lt_upper, sloped_sr_lines, sloped_sr_lines_starts