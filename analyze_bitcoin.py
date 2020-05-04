from Modules.analyzePlotManual import get_analyzed_plot
#import os

def main():
   #if not os.path.exists(r'./Datasets'):
   #   from Modules.generateDatasetsOanda import main as gen_datasets
   #   gen_datasets()

   instruments = ['BTC_USD']
   timePeriods = ['D', 'H1']
   windows = [[550, 120], [550, 300], [1375, 300]]

   for instr in instruments:
      for time in timePeriods:
         for window in windows:
            print(f"Analyzing {instr} {time} in window {window[0]}-{window[1]}")
            get_analyzed_plot(instr, time, len_longterm=window[0], len_window=window[1], txtOutput=False)

   # auto-open output text files in Sublime Text? Maybe dependent on specifically desired currencies/timeframes to analyze

if __name__ == '__main__':
   print('')
   main()