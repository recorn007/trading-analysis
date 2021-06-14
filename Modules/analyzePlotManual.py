import pandas as pd
import numpy as np
from Modules.features import new_datetime_complete
from Modules.candlePlotting import plot_ticks


def split_df(instr, period, len_longterm, len_window, normalize_to_pip=True):
    #df = get_df_deltas(instr, period)
    df = pd.read_csv(r'Datasets/{} {}.csv'.format(instr, period), parse_dates=[0], index_col=0)
    df = df.sort_values(by='time')

    if normalize_to_pip: # convert to pips from $n.nn
        for col in ('Open', 'High', 'Low', 'Close'):
            df[col] = df[col] / 1000

    min_w_lt = max(0, len(df)-(len_longterm+1))
    min_w = max(0, len(df)-(len_window+1))
    
    # execute new_datetime on penultimate candle for analysis to wait til close of the candle to make proper IPDE's
    max_w = len(df)-2
    
    # remove Rejection price on points leading up to df_window
    df['Rejection'].iloc[:min_w+1] = np.nan

    df_longterm = df.iloc[min_w_lt:max_w, :].copy()
    df_window = df.iloc[min_w:max_w, :].copy()
    df_lastclosed = df.iloc[max_w].copy()
    df_last = df.iloc[-1].copy()
    
    df_window = df_window_cols(df_window)

    # Save df to local csv
    #while len(df) > len_longterm:
    #    df = df.iloc[1:]
    #df.to_csv(r'Datasets/{} {}.csv'.format(instr, period))

    return df_longterm, df_window, df_lastclosed, df_last


def df_window_cols(df_window):
    cols = ['Volume', 'Candle Pattern', 'Same-sized Candle Trend Rejection', 'Engulfing Pattern', 'Immediate Trend Direction', 'Rejection',
            'Near Short-term Control', 'Near Long-term Control', 'In Excess Above Short-term Value', 'In Excess Below Short-term Value',
            'In Excess Above Long-term Value', 'In Excess Below Long-term Value', 'Rejected Short-term Control', 'Rejected Long-term Control',
            'Rejected Short-term Upper Limit', 'Rejected Short-term Lower Limit', 'Rejected Long-term Upper Limit', 'Rejected Long-term Lower Limit',
            'Near minor SR', 'Near major SR', 'Near Sloped SR', 'Rejected minor SR line', 'Rejected major SR line',
            'Rejected Sloped SR line', 'Long-term Trend Direction', 'In Excess of Long-term Value Area, Trend-following',
            'In Excess of Long-term Value Area, Counter-trend', 'Rejected Fibo level 236', 'Rejected Fibo level 382', 'Rejected Fibo level 618',
            'Closed above previous (green)', 'Close below previous (red)']
    for c in cols:
        if c != 'Rejection': df_window[c] = np.nan
    df_window = df_window[['Open', 'High', 'Low', 'Close'] + cols]
    
    return df_window


def get_analyzed_plot(instr, period, len_longterm, len_window, txtOutput=True):
    df_longterm, df_window, df_lastclosed, df_last = split_df(instr, period, len_longterm, len_window)
    df_longterm, df_window, shortterm_SR, longterm_SR, shortterm_trend, st_lower, st_upper, longterm_trend, lt_lower, lt_upper, sloped_sr_lines, sloped_sr_lines_starts \
        = new_datetime_complete(df_longterm, df_window, df_lastclosed, keep_df_size=(len(df_longterm) > len_longterm))
    len_of_future_bars = 50
    df_window_plt = df_window.iloc[1:].append(df_last)

    # df_window_plt = df_window_plt.reindex(df_window_plt.index.append(
    #     pd.date_range(df_window_plt.index[-1], periods=len_of_future_bars-2, closed='right',
    #                   freq='{}{}'.format(period[1] if len(period)>1 else '1', period[0] if len(period)>1 else period))))
    # modify the format for freq in above if considering getting candles with granularities other than 1 or 4
    
    df_window_plt = df_window_plt.reindex(df_window_plt.index.append(pd.Index(str(n) for n in range(len_of_future_bars-2))))
    #raise Exception(len(df_window_plt))
    # for i in range(len(df_window_plt)-55, len(df_window_plt)-45): print(df_window_plt.index[i])

    plot_ticks(df_window_plt, longterm_SR, shortterm_SR,
               longterm_trend.reindex(df_window.index, axis=0),
               lt_lower.reindex(df_window.index, axis=0), lt_upper.reindex(df_window.index, axis=0),
               shortterm_trend, st_lower, st_upper,
               sloped_sr_lines, sloped_sr_lines_starts, df_last.name, len_of_future_bars,
               instr, period, len_longterm, len_window, txtOutput)

    return