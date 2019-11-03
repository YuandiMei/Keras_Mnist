from keras.datasets import mnist  
from model import *
from sklearn.model_selection import train_test_split
import random
from keras.callbacks import ModelCheckpoint,Callback,ReduceLROnPlateau



n_class, n_len, width, height = 11, 5, 28, 28

def loading_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train,y_train), (x_test,y_test)

def treat_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    x_raw_train,x_raw_valid, y_raw_train, y_raw_valid = train_test_split(x_train,y_train, test_size=0.1, random_state=50)
    return x_raw_train,x_raw_valid, y_raw_train, y_raw_valid, x_test, y_test


def generate_dataset(X, y):     

    X_len = X.shape[0] # 原数据集有几个，新数据集还要有几个 
    # 新数据集的shape为(X_len, 28, 28*5, 1)，X_len是X的个数，原数据集是28x28，
    # 取5个数字(包含空白)拼接，则为28x140, 1是颜色通道，灰度图，所以是1    
    X_gen = np.zeros((X_len, height, width*n_len, 1), dtype=np.uint8)   
    # 新数据集对应的label，最终的shape为（5,  X_len，11）    
    y_gen = [np.zeros((X_len, n_class), dtype=np.uint8) for i in range(n_len)]         
    for i in range(X_len):        # 随机确定数字长度         
        rand_len = random.randint(1, 5)         
        lis = list()        # 设置每个数字         
        for j in range(0, rand_len):            # 随机找一个数             
            index = random.randint(0, X_len - 1)
            #将对应的y置1, y是经过onehot编码的，所以y的第三维是11，0~9为10个数字，10为空白，哪个索引为1就是数字几             
            y_gen[j][i][y[index]] = 1             
            lis.append(X[index].T)        # 其余位取空白             
        for m in range(rand_len, 5):            # 将对应的y置1             
            y_gen[m][i][10] = 1
            lis.append(np.zeros((28, 28),dtype=np.uint8))    
        lis = np.array(lis).reshape(140,28).T    
        X_gen[i] = lis.reshape(28,140,1)     
    return X_gen, y_gen



def generate_all_data():
    x_raw_train,x_raw_valid, y_raw_train, y_raw_valid, x_test, y_test=treat_data()
    X_train, Y_train = generate_dataset(x_raw_train, y_raw_train)
    X_valid, Y_valid = generate_dataset(x_raw_valid, y_raw_valid)
    X_test , Y_test  = generate_dataset(x_test,      y_test     )   
    
    #显示前15个生成的图片
    for i in range(15):  
        plt.subplot(5, 3, i+1)
        index = random.randint(0, n_test-1) 
        title = ''     
        for j in range(n_len):   
            title += str(np.argmax(Y_test[j][index])) + ','     
        plt.title(title)
    plt.imshow(X_test[index][:,:,0], cmap='gray') 
    plt.axis('off')
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test 

def train_multi_model(model,model_name_h5,epochs=10):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test=generate_all_data()
    
    learnrate_reduce_1 = ReduceLROnPlateau(monitor='val_dense_2_acc', patience=2, verbose=1,factor=0.8, min_lr=0.00001) 
    learnrate_reduce_2 = ReduceLROnPlateau(monitor='val_dense_4_acc', patience=2, verbose=1,factor=0.8, min_lr=0.00001) 
    learnrate_reduce_3 = ReduceLROnPlateau(monitor='val_dense_6_acc', patience=2, verbose=1,factor=0.8, min_lr=0.00001) 
    learnrate_reduce_4 = ReduceLROnPlateau(monitor='val_dense_8_acc', patience=2, verbose=1,factor=0.8, min_lr=0.00001) 
    learnrate_reduce_5 = ReduceLROnPlateau(monitor='val_dense_10_acc',patience=2, verbose=1,factor=0.8, min_lr=0.00001)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=200, validation_data=(X_valid, Y_valid), 
              callbacks=[learnrate_reduce_1,learnrate_reduce_2,learnrate_reduce_3,learnrate_reduce_4,learnrate_reduce_5])

    result = model.evaluate(np.array(X_test).reshape(len(X_test),28,140,1), 
                            [Y_test[0], Y_test[1], Y_test[2], Y_test[3], Y_test[4]], batch_size=32)   
    model.save(model_name_h5) #保存模型
    res=result[6] * result[7] * result[8] * result[9] * result[10] 
    print('The model\'s accuracy is '+str(res))
    return result
    

def reshaping_data(x_train, y_train, x_test, y_test):
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1] ,x_train.shape[2],1) 
    X_test  =  x_test.reshape(x_test.shape[0], x_test.shape[1] , x_test.shape[2],1)  
    Y_train = (np.arange(10) == y_train[:, None]).astype(int) 
    Y_test  = (np.arange(10) ==  y_test[:, None]).astype(int)
    return (X_train,Y_train), (X_test,Y_test)

def training_model(model, X_train, Y_train, batch_size=100, epochs=20,shuffle=True, validation_split=0.1, verbose=1):
    model.fit(X_train,Y_train,batch_size,epochs,shuffle,validation_split,verbose)
    
def testing_model(model,X_test,Y_test,batch_size=200,verbose=1):
    print("test set")
    scores = model.evaluate(X_test,Y_test,batch_size=200,verbose=1)
    print("")
    print("The test loss is %f" % scores)
    return scores

def model_accuracy(model,X_test,Y_test):    
    result = model.predict(X_test,batch_size=200,verbose=2)
    result_max = np.argmax(result, axis = 1)
    test_max = np.argmax(Y_test, axis = 1)
    result_bool = np.equal(result_max, test_max)
    true_num = np.sum(result_bool)
    print("")
    print("The accuracy of the model is %f" % (true_num/len(result_bool)))
    return (true_num/len(result_bool))


