from Modules.analyzePlotOanda import get_analyzed_plot
import os

def main():
   if not os.path.exists(r'./Datasets'):
      from Modules.generateDatasetsOanda import main as gen_datasets
      gen_datasets()

   currency = ['EUR_USD', 'EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'AUD_USD', 'USD_CAD', 'USD_JPY']
   timePeriods = ['M', 'W', 'D', 'H4', 'H1']
   #windows = [[550, 120], [550, 300], [1375, 300]]

   for cur in currency:
      for time in timePeriods:
         print("Analyzing {} {}".format(cur, time))
         get_analyzed_plot(cur, time, len_longterm=550, len_window=120, txtOutput=False)

   # auto-open output text files in Sublime Text? Maybe dependent on specifically desired currencies/timeframes to analyze

if __name__ == '__main__':
   print('')
   main()