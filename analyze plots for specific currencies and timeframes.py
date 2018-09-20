from MyModules.analyzePlotOanda import get_analyzed_plot

def main():
    #currensy = ('EUR_USD', 'EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'AUD_USD', 'USD_CAD', 'USD_JPY')
    currensy = ('EUR_JPY', 'USD_JPY') # problems
    timePeriods = ('M', 'W', 'D', 'H4')

    for cur in currensy:
        for time in timePeriods:
            print(cur, time)
            get_analyzed_plot(cur, time, len_longterm=550, len_window=120)

    # auto-open output text files in Sublime Text? Maybe dependent on specifically desired currencies/timeframes to analyze

if __name__ == '__main__':
    main()