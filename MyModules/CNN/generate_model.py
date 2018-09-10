import pandas as pd
import numpy as np
#import matplotlib.finance as finplt
import mpl_finance as finplt
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from os import listdir
import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv(r'/home/michael/repos/the-forex-ai/datasets/dukascopy - EURUSD_Candlestick_4_Hour_BID_31.12.2015-30.12.2016.csv',
                     parse_dates=[0], index_col=0, date_parser=lambda d: pd.datetime.strptime(d[:13], '%d.%m.%Y %H'))

    df_window = df.iloc[40:, :]

    y = pd.read_csv(r'y candles.csv',
                    parse_dates=[0], index_col=0, date_parser=lambda d: pd.datetime.strptime(d[:13], '%Y-%m-%d %H'))

    ind = [d in y.index for d in df_window.index]
    df_window = df_window.loc[ind]

    # #gen_img_files(df_window) # comment out if images files are already in ./candles
    # X_train, y_train = conv_img_files_to_array(y)
    # model = let_us_have_a_CNN(X_train, y_train)
    # model.save(r'An eye for an eye - a CNN model.h5')  # creates an HDF5 file 'my_model.h5' # Save dat model

    from keras.models import load_model
    model = load_model('An eye for an eye - a CNN model.h5')
    test_CNN(df, model)

    return

def gen_img_files(df_window):
# Save plots for individual candlesticks in window
    plt.rcParams['figure.figsize'] = (0.4, 0.8)

    for i in range(len(df_window)):
        try:
            fig, ax = plt.subplots()
            plt.rcParams['figure.figsize'] = (0.4, 0.8)
            to_plot = df_window[i:i+1]
            finplt.candlestick2_ohlc(ax, to_plot.Open, to_plot.High, to_plot.Low, to_plot.Close,
                                 width=0.6, colorup='g', colordown='r', alpha=1)
            plt.axis('off')
            plt.savefig('./candles/' + str(df_window.iloc[i].name)[:-6] + 'h.jpg')
            plt.close()
        except:
            continue
            
# Convert plots to greyscale and Keras-ready
    files = listdir('./candles/')
    files.sort()

    for file in files:
        i = Image.open('./candles/' + file).convert('L')
        j = np.asarray(i.getdata(), dtype=np.float64).reshape((i.size[1], i.size[0]))
        j = np.asarray(j, dtype=np.uint8) #if values still in range 0-255! 
        img = Image.fromarray(j, mode='L')
        img.save('./candles/' + file)

    return

def conv_img_files_to_array(y):
# Define and preprocess X multi-dimentional array of all images
    files = listdir('./candles/')
    files.sort()

    X = [[] for _ in range(len(files))]
    for i, file in enumerate(files):
        X[i].append(np.array(Image.open('./candles/' + file)))

    X = np.array(X)
    X = X.astype('float32')
    X /= 255

    X = X.reshape(X.shape[0], 40, 80, 1)
    X.shape

    y = to_categorical(y['Category'])

    X_train = X
    y_train = y

    return X_train, y_train

def let_us_have_a_CNN(X_train, y_train):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(40, 80, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=25, verbose=1)

# For a test split, define bigger X_test set from df, and generate images for them. Predict their shape, and save images with text of what is predicted
    return model

def test_CNN(df, model):
    df_test = df.iloc[1000:1200, :]
# Save plots for individual candlesticks in window
    plt.rcParams['figure.figsize'] = (0.4, 0.8)

    for i in range(len(df_test)):
        try:
            fig, ax = plt.subplots()
            plt.rcParams['figure.figsize'] = (0.4, 0.8)
            to_plot = df_test[i:i+1]
            finplt.candlestick2_ohlc(ax, to_plot.Open, to_plot.High, to_plot.Low, to_plot.Close,
                                 width=0.6, colorup='g', colordown='r', alpha=1)
            plt.axis('off')
            plt.savefig('./test/' + str(df_test.iloc[i].name)[:-6] + 'h.jpg')
            plt.close()
        except:
            continue
            
    test_files = listdir('./test/')
    test_files.sort()

# Convert plots to greyscale and Keras-ready
    for file in test_files:
        i = Image.open('./test/' + file).convert('L')
        j = np.asarray(i.getdata(), dtype=np.float64).reshape((i.size[1], i.size[0]))
        j = np.asarray(j, dtype=np.uint8) #if values still in range 0-255! 
        img = Image.fromarray(j, mode='L')
        img.save('./test/' + file)
        
# Define and preprocess X multi-dimentional array of all images
    X_test = [[] for _ in range(len(test_files))]
    for i, file in enumerate(test_files):
        X_test[i].append(np.array(Image.open('./test/' + file)))

    X_test = np.array(X_test)
    X_test = X_test.astype('float32')
    X_test /= 255
    X_test = X_test.reshape(X_test.shape[0], 40, 80, 1)
    print(X_test.shape)

    y_test = model.predict(X_test)

# Write model pred onto image files
    cat = {1: 'Hammer with body near high', 2: 'Hammer with body near low', 3: 'Spinning top',
           4: 'Doji with close near high', 5: 'Doji with close near low', 6: 'Doji with close near middle',
           7: 'Marubozu', 0: 'No category'}

    text = []
    for i in range(len(y_test)):
        pred = cat[np.argmax(y_test[i])]
        text = np.append(text, pred)

    for f, file in enumerate(test_files):
        i = Image.open('./test/' + file)
        j = Image.fromarray(np.full((200, 193), 255, dtype='uint8'))
        
        basewidth=95
        wpercent = (basewidth/float(i.size[0]))
        hsize = int((float(i.size[1])*float(wpercent)))
        i = i.resize((basewidth,hsize), Image.ANTIALIAS)
        j.paste(i, (0,0))
        
        draw = ImageDraw.Draw(j)
        draw.text((0, 0), text[f], (0), ImageFont.truetype("OpenSans-Regular.ttf", 14))
        j.save('./test_result/' + file)

    return

if __name__ == '__main__':
	main()