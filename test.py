# Need to rebuild CNN with scripts in order to achieve one-liner get_analyzed_plot

from MyModules.analyzePlotOanda import get_analyzed_plot

get_analyzed_plot('EUR_USD', '1', 'D', len_longterm=550, len_window=120)