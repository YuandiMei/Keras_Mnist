import skimage.io
import cv2
from PIL import Image as Im

from model import *
from train import *
from utils import *

n_class, n_len, width, height = 11, 5, 28, 28

def show_image(x):
    plt.imshow(x)
    
def predict(model,x):
    X=x.reshape(1,28,28,1) 
    result = model.predict(x,batch_size=1,verbose=2)
    return result

def ImageToMatrix(filename,width=28,height=28):
    # 读取图片
    im = Im.open(filename)
    # 显示图片
    #plt.imshow(im) 
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    new_data = np.reshape(data,(height,width))
    dat=cv2.resize(new_data,(28,28))
    dat=dat.astype(int)
    
    return dat


def merge_5_images(image1='screen.jpg',image2='screen.jpg',image3='screen.jpg',image4='screen.jpg',image5='screen.jpg'):
    da=[]
    da.append(ImageToMatrix(image1))
    da.append(ImageToMatrix(image2))
    da.append(ImageToMatrix(image3))
    da.append(ImageToMatrix(image4))
    da.append(ImageToMatrix(image5))
    
    X_out=np.zeros((28,140))

    for i in range(5):
        i_1=28*i
        i_2=28*(i+1)
        X_out[:,i_1:i_2]=da[i]
    
    return X_out
    

#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def Prediction(model,data):
    arr = np.array(data).reshape((28,28,1))
    arr = np.expand_dims(arr, axis=0)
    prediction = model.predict(arr)[0]
    bestclass = ''
    bestconf = -1
    for n in [0,1,2,3,4,5,6,7,8,9]:
        if (prediction[n] > bestconf):
            bestclass = str(n)
            bestconf = prediction[n]
    print ('Predicted the digit ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')


def RunPredict(x):
    model=loading_model('my_model_architecture.json','my_model_weights.h5')
    data=ImageToMatrix(x)
    plt.imshow(data)
    plt.show()
    Prediction(model,data)
    
def get_result(result):
   # 将 one_hot 编码解码
    resultstr = ''
    for i in range(n_len):
       resultstr += str(np.argmax(result[i])) + ','
    return resultstr


def check_result(model,X_test,Y_test):
    n_test=len(Y_test[0])
    index = random.randint(0, n_test-1)
    y_pred = model.predict(X_test[index].reshape(1,28,140,1))

    plt.title('real: %s\npred:%s'%(get_result([Y_test[x][index] for x in range(n_len)]), get_result(y_pred)))
    plt.imshow(X_test[index,:,:,0], cmap='gray')

    plt.axis('off')

def predict_multiple_images(model,image1='screen.jpg',image2='screen.jpg',image3='screen.jpg',image4='screen.jpg',image5='screen.jpg'):
    X=merge_5_images(image1,image2,image3,image4,image5)
    Y=model.predict(X.reshape(1,28,140,1))
    plt.title('pred:%s'%(get_result(Y)))
    #plt.imshow(X, cmap='gray')
    plt.imshow(X)
    
def predict_multiple_from_X_test(i,model,X_test):
    Y=model.predict(X_test[i,:,:,0].reshape(1,28,140,1))
    plt.title('pred:%s'%(get_result(Y)))
    plt.imshow(X_test[i,:,:,0], cmap='gray')

def run_predict_multiple_from_X_test(i):
    model = load_model('multiple_mnist_model.h5')  # 载入模型
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    x_raw_train,x_raw_valid, y_raw_train, y_raw_valid, x_test, y_test=treat_data()
    X_test,Y_test=generate_dataset(x_test,y_test)   
    predict_multiple_from_X_test(i,model,X_test)    
    
    