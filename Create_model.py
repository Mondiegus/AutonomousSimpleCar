# -*- coding: utf-8 -*-
"""Copy of Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u1m8q7kx7GJzLuPzntPqKA2dKQwbmFCi
"""

!nvidia-smi

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
!pip install imgaug;;
from imgaug import augmenters as iaa;
from imgaug import parameters as iap;
import random;
from sklearn.model_selection import train_test_split;

!pip install tensorflow==1.13.1;
!pip install torch==1.0.1.post2;
!pip install torchvision==0.2.1;
!pip install --upgrade keras;

import tensorflow.python.keras.losses;
from tensorflow.python.keras.models import Sequential;
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, Lambda;
from tensorflow.python.keras.losses import categorical_crossentropy;
from tensorflow.python.keras.losses import categorical_hinge;
from tensorflow.python.keras.losses import cosine;
from tensorflow.python.keras.losses import cosine as cosine_proximity;
from tensorflow.python.keras.losses import deserialize;
!pip install livelossplot;
from livelossplot import PlotLossesKeras;

y1 = np.load(fr'drive/My Drive/Y0.npy')
y2 = np.load(fr'drive/My Drive/Y1.npy')
y3 = np.load(fr'drive/My Drive/Y2.npy')
y4 = np.load(fr'drive/My Drive/Y3.npy')
y5 = np.load(fr'drive/My Drive/Y4.npy')
y6 = np.load(fr'drive/My Drive/Y5.npy')
y7 = np.load(fr'drive/My Drive/Y6.npy')
y8 = np.load(fr'drive/My Drive/Y7.npy')
y9 = np.load(fr'drive/My Drive/Y8.npy')
x1 = np.load(fr'drive/My Drive/X0.npy')
x2 = np.load(fr'drive/My Drive/X1.npy')
x3 = np.load(fr'drive/My Drive/X2.npy')
x4 = np.load(fr'drive/My Drive/X3.npy')
x5 = np.load(fr'drive/My Drive/X4.npy')
x6 = np.load(fr'drive/My Drive/X5.npy')
x7 = np.load(fr'drive/My Drive/X6.npy')
x8 = np.load(fr'drive/My Drive/X7.npy')
x9 = np.load(fr'drive/My Drive/X8.npy')

y = np.concatenate((y1, 
                    y2, 
                    y3, 
                    y4, 
                    y5, 
                    y6, 
                    y7, 
                    y8, 
                    y9), axis=0)
X = np.concatenate((x1, 
                    x2, 
                    x3, 
                    x4, 
                    x5, 
                    x6, 
                    x7, 
                    x8, 
                    x9), axis=0)

from google.colab import drive
drive.mount('/content/drive')

print(X.shape, y.shape)

plt.hist(y,50)

def augment(img, steering_angle):
  # Flip - odbicie lustrzane
  if random.random() > 0.5:
    img = img[:, ::-1, :]
    steering_angle = -steering_angle
  #blur - rozmazanie
  blurer = iaa.GaussianBlur(iap.Uniform(0.1, 1.0))
  img = blurer.augment_image(img)
  #shuffle
  ColorShuffle = iaa.ChannelShuffle(p=0.7)
  img = ColorShuffle.augment_image(img)
  #SuperPixels
  superpixel = iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
  img = superpixel.augment_image(img)
  #Fog
  Clouds = iaa.Clouds()
  img = Clouds.augment_image(img)
  #Snowflakes
  # Snowflakes = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))
  # img = Snowflakes.augment_image(img)
  #Translate
  tx = random.randint(-20,20)
  translater = iaa.Affine(translate_px = {"x":tx}, mode = 'edge')
  img = translater.augment_image(img)
  steering_angle += tx*0.02
  
  return img, steering_angle

plt.imshow(X[0])
print(y[0])
plt.show()

img2, angle = augment(X[0],y[0])

plt.imshow(img2)
print(angle)
plt.show()

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

Xgen = []
Ygen = []
for i in range(y_train.shape[0]):
  img, angle = augment(X[i],y[i])
  Xgen.append(img)
  Ygen.append(angle)
  
Xgen = np.array(Xgen)
Ygen = np.array(Ygen)
print(Xgen.shape, Ygen.shape)

plt.hist(Ygen,50, facecolor = 'red')
plt.hist(y_train,50, facecolor = 'green')

def myModel(input_shape):
  model = Sequential([
      Lambda(lambda x: (x-128)/255, input_shape = input_shape, name = "normalize"),
      Conv2D(64, (3,3), activation = 'relu'),
      #MaxPooling2D(pool_size=(2,2)),
      Conv2D(32, (3,3), activation = 'relu'),
      MaxPooling2D(pool_size=(2,2)),
      Conv2D(16, (3,3), activation = 'relu'),
      MaxPooling2D(pool_size=(2,2)),
      Conv2D(8, (3,3), activation = 'relu'),
      MaxPooling2D(pool_size=(2,2)),
      Dropout(rate = 0.5),
      Flatten(),
      Dense(512, activation = 'relu'),
      Dropout(rate = 0.3),
      Dense(512, activation = 'relu'),
      Dropout(rate = 0.3),
      Dense(16, activation = 'relu'),
      Dense(1),
  ])
  return model

model = myModel(X[0].shape)
model.summary()
model.compile('adam' ,'mse')

X[0].shape

batch_size = 256
epochs = 15

model.fit(Xgen,Ygen, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks = [PlotLossesKeras()])

ypred = model.predict(X)
plt.plot(y, 'g',ypred,'r')
plt.show()

model.save(fr'mymodel1.h5')

!pip install folium==0.2.1
!pip install imgaug==0.2.7
#!pip install autokeras

!pip uninstall pytorch torchvision -y
!pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-win_amd64.whl
!pip3 install torchvision



import autokeras as ak

model_ak = ak.ImageRegressor(verbose = True)
model_ak.fit(Xgen,Ygen, time_limit=60*60)

score = model_ak.evaluate(X_val, y_val)
print(score)

ypred = model_ak.predict(X)

plt.plot(y,'g', ypred,'r')
plt.show()

model_ak.export_autokeras_model("autoKeras.pkl")

from autokeras.utils import pickle_from_file

model = pickle_from_file('autoKeras.pkl')
import keras

yp = model.predict(X)

model.save("autokeras__.h5")