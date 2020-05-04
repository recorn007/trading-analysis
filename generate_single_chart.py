from Modules.analyzePlotOanda import get_analyzed_plot
import sys

def main():
   get_analyzed_plot(sys.argv[1], sys.argv[2], len_longterm=550, len_window=120)

if __name__ == '__main__':
   main()