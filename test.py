# Need to rebuild CNN with scripts in order to achieve one-liner get_analyzed_plot

from MyModules.analyze_plot_oanda import get_analyzed_plot

get_analyzed_plot('EUR_USD', '1', 'D', len_longterm=551, len_window=121)