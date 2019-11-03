import keras as ks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import *
from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.callbacks import ModelCheckpoint,Callback
from keras import backend as K
from keras.utils.vis_utils import plot_model


def number(i):
    plt.imshow(x_raw_train[i])
    plt.show()
    print(y_raw_train[i])


def visualizing(model,file_name_jpg):
    #可视化Visualizing
    plot_model(model, to_file=file_name_jpg)
    

def opti_loss(model,optimizer=SGD,loss_function='categorical_crossentropy', lr=0.01,decay=1e-6, momentum=0.9, nesterov=True):
    #optimizer=SGD,Adadelta, Adagrad, RMSprop
    optim = optimizer(lr, decay, momentum, nesterov) # optimizer  
    model.compile(loss=loss_function, optimizer=optim) # loss function
    
    #一些参数：
    #batch_size：对总的样本数进行分组，每组包含的样本数量
    #epochs ：训练次数
    #shuffle：是否把数据随机打乱之后再进行训练
    #validation_split：拿出百分之多少用来做交叉验证
    #verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
    
def verify_number(x,y,i):
    print("We'll verify the "+str(i)+"e number of the train data:")
    plt.imshow(x[i])
    plt.show()
    print(y[i])