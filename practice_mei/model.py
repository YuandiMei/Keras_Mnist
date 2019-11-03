import keras as ks
import pandas as pd
import numpy as np
import h5py
import gzip
import struct
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential,Model,load_model
from keras.models import model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import  BatchNormalization
from IPython.display import Image, SVG
from keras.utils.vis_utils import plot_model,model_to_dot
from keras.layers.advanced_activations import PReLU

n_test=10000
def cnn_modeling( ): 
    model = ks.models.Sequential()
    #第一层：卷积
    model.add(Conv2D(68, kernel_size=(3, 3),activation='tanh',input_shape=(28,28,1))) 
    #tensorflow ---> channels_last --->(28,28,1)
    #第二层：卷积池化
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #输出层：全连接
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    return model

def cnn_multoutput():

    inputs = Input(shape=(28, 140, 1))

    conv_11 = Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu')(inputs)
    max_pool_11 = MaxPool2D(pool_size=(2,2))(conv_11)
    dropout_11=Dropout(0.25)(max_pool_11)
    
    conv_12 = Conv2D(filters= 11, kernel_size=(3,3), padding='Same', activation='relu')(dropout_11)
    max_pool_12 = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv_12)
    dropout_12=Dropout(0.25)(max_pool_12)

    flatten11 = Flatten()(dropout_12)
    
    hidden11 = Dense(15, activation='relu')(flatten11)
    prediction1 = Dense(11, activation='softmax')(hidden11)

    hidden21 = Dense(15, activation='relu')(flatten11)
    prediction2 = Dense(11, activation='softmax')(hidden21)

    hidden31 = Dense(15, activation='relu')(flatten11)
    prediction3 = Dense(11, activation='softmax')(hidden31)

    hidden41 = Dense(15, activation='relu')(flatten11)
    prediction4 = Dense(11, activation='softmax')(hidden41)

    hidden51 = Dense(15, activation='relu')(flatten11)
    prediction5 = Dense(11, activation='softmax')(hidden51)

    model = Model(inputs=inputs, outputs=[prediction1,prediction2,prediction3,prediction4,prediction5])

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def saving_model(model):
    #保存神经网络结构以及训练好的参数
    json_string=model.to_json()
    open('my_model_architecture.json','w').write(json_string)
    model.save_weights('my_model_weights.h5')
    

def loading_model(name_model_architecture_json,name_model_weights_h5):
    #加载模型
    model=model_from_json(open(name_model_architecture_json).read())
    model.load_weights(name_model_weights_h5)
    return model
    
    
    








