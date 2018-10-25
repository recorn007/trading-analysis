from MyModules.analyzePlotOanda import get_analyzed_plot
import os

def main():
   print('')
   
   if not os.path.exists(r'./Datasets'):
      from MyModules.generateDatasets import main as gen_datasets
      gen_datasets()

   currensy = ('EUR_USD', 'EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'AUD_USD', 'USD_CAD', 'USD_JPY')
   timePeriods = ('M', 'W', 'D', 'H4')

   for cur in currensy:
      for time in timePeriods:
         print("Analyzing {} {}".format(cur, time))
         get_analyzed_plot(cur, time, len_longterm=550, len_window=120, txtOutput=False)

   # auto-open output text files in Sublime Text? Maybe dependent on specifically desired currencies/timeframes to analyze

if __name__ == '__main__':
   main()