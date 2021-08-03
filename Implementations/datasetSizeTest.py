import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle

X=pickle.load(open('eyeFeatures.pickle','rb'))
y=pickle.load(open('eyeLabels.pickle','rb'))

X=X/255.0 

##Prepare test sets###
testX=X[80000:]
testY=y[80000:]

### set varying sizes for dataset ###
sizes=[10000,20000,30000,40000,50000,60000,70000,80000]
data=[]

for s in sizes:
    print("##### Training "+str(s)+" train set model #####")
    sampleX=X[:s]
    sampleY=y[:s]

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(50,50,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Dropout(0.05))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #final_model=model.fit(sampleX,sampleY, batch_size=32, epochs=10, validation_split=0.3)
    model.fit(sampleX,sampleY, batch_size=32, epochs=10, validation_split=0.3)
    model_metrics=model.evaluate(testX, testY)
    data.append([s,model_metrics[0],model_metrics[1]])

df = pd.DataFrame(data, columns = ['Train Set Size','Loss', 'Accuracy'])
filename="datasetSizeTest3.csv"
df.to_csv(filename)
