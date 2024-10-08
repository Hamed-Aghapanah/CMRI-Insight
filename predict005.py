"""   Created on Fri May 26 10:03:14 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""



# def function_predictor1(img1,mask1):
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os 
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from Models_unet import get_model_MSTGANet_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from model_CardSegNet2 import BBVDAMLF_model 
from saed import masker
from saed import Grad_CAM1
from saed import Grad_CAM2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img    
from skimage.transform import resize
from keras.models import load_model
import cv2
from models_RASTI_CE_Net  import get_model as get_model_CE_Net
from models_RASTI_NN_Unet import nnUNet_2D as get_model_NN_Unet
import keras
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.layers  import Reshape
from keras.layers import BatchNormalization , Activation, Dropout
# from keras.layers import  BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate as  concatenate
from keras.layers import UpSampling2D  , AveragePooling2D
import matplotlib.pyplot as plt
import keras
from keras.utils.layer_utils import count_params
from math import isnan,isinf
import copy
from tensorflow.keras.constraints import max_norm

from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers.core import Activation, SpatialDropout2D
from DAttention import Channel_attention as CAM, Position_attention as PAM
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img    

import time
import datetime# import jdatetime# timee =jdatetime.date.today()
import metricss
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
# from tensorflow.keras.applications.
import numpy as np
import pandas as pd
# from  saed import *
import keras as K
from skimage.transform import resize
import os
import time
time0 = time.time() 

grad_cam_enable = False
# =============================================================================
#  def
# =============================================================================
def roundd(a=123.456,b=1):  # a=0.123456789;   # b=3
    aa = a*100*10**b;
    aa= np.int16 (aa);
    aa=np.fix(aa)
    aa=int(a)
    aa=aa/(10**b)
    return aa
    
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    try:
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dicee=(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    except:
        
        try:
            y_pred = y_pred.numpy()
            y_true = y_true.numpy()
        except:1    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    under =(K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    dicee=(2. * intersection + smooth) / under
    
    
    # saed
    # dicee=(2. * np.mean(intersection) + smooth) / np.mean(under)
    # print('dicee = ',dicee)    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dicee = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return dicee 

def ther (a,th):
            import copy
            th2 = (th +np.max(a)*1) /2
            c=copy.deepcopy(a*0)
            for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    if a[i,j] ==0:
                        c[i,j]=0
                    if a[i,j] >=th2 and th2>0:
                        c[i,j]=1
            return c
def max_segman (image):
    import cv2
    import imutils
    import numpy as np
    cv2.destroyAllWindows()
    
    image=np.max(image)-image
    cv2.waitKey(500)
    gray_scaled = image[:,:];#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("gray_scaled", gray_scaled)
    thresh = cv2.threshold(gray_scaled, 225,225, cv2.THRESH_BINARY_INV)[1]
    # sss
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    area=[];
    for contour in contours:# loop over each contour found
        output = 0*image.copy()
        
        cv2.drawContours(output, [contour], -1,(255,255,255),thickness=cv2.FILLED) # outline and display them, one by one.
        area.append(np.sum(output))
        #

    index = np.where (area==np.max(area))
    output = 0*image.copy()
    cv2.drawContours(output, [contours[index[0][0]]], -1,(255,255,255),thickness=cv2.FILLED) # outline and display them, one by one.
    
    return output
    
def erosioning(a,pixels):
    import copy
    c=copy.deepcopy(a)
    for i in range(2,np.shape(a)[0]-2):
        for j in range(2,np.shape(a)[1]-2):
            if a[i,j]==0:
                if np.sum (a[i-1:i+1,j-1:j+1] ) <=pixels:
                    c[i,j]=0
    return c

def smoothing(a,pixels):
    import copy
    c=copy.deepcopy(a)
    for i in range(2,np.shape(a)[0]-2):
        for j in range(2,np.shape(a)[1]-2):
            if a[i,j]==0:
                if np.sum (a[i-1:i+1,j-1:j+1] ) >=pixels:
                    c[i,j]=1
    return c
def dot_add(a=123.456,b=1):
            import copy
            c=copy.deepcopy(a*0)
            for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    c[i,j]=a[i,j]+b[i,j]               
            return c
def dot_multi(a=123.456,b=1):
    import copy
    c=copy.deepcopy(a*0)
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            c[i,j]=a[i,j]*b[i,j]
    return c
def community(a=123.456,b=1):
    import copy
    c=copy.deepcopy(a*0)
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            c[i,j]=a[i,j]+b[i,j]
            if c[i,j]>1:
                c[i,j]=1
    return c
    
def sumer(a=123.456 ):
    c=np.sum(a)
    c=np.sum(c)
    c=np.sum(c)
    c=int(c)
    return c

def dice_function (Y,YY1):           
            d_r=1; d_l=1;      d_m=1;d_b=1;
            from metricss import dice_coef
            d_r=0.0001*int(np.mean(10000*dice_coef(Y[:,:,0], YY1[:,:,0], smooth=1)))
            d_m=0.0001*int(np.mean(10000*dice_coef(Y[:,:,1], YY1[:,:,1], smooth=1)))
            d_l=0.0001*int(np.mean(10000*dice_coef(Y[:,:,2], YY1[:,:,2], smooth=1)))
            d_overal = ( (d_l+d_r+d_m+d_b))/4
         
            return d_overal,d_r,d_l,d_b,d_m 



#  VARIABLES
DICE_ALL=[]



GRAD_CAM_FOLDER=2
plt.close("all")

GRAD_CAM_FOLDER='GRAD_CAM_FOLDER'
try:os.mkdir(GRAD_CAM_FOLDER)
except:s=1

just_one=1
indexes=range(0,6)
indexes=range(19,21)
# indexes=range(0,950)


im_width=128;im_height=128

# file_name   = 'input3_'+str(im_width)+'_'+str(im_height)+'.npz'
# data1 = np.load(file_name ,allow_pickle=True )
# X  = data1['X']
# XXX = np.load('X3.npz', allow_pickle = True)
# X3 = XXX['X3']
# Y1 = data1['Y1']
# Y2 = data1['Y2']
# Y3 = data1['Y3']
# np.savez('final_data.npz', X = X, Y1=Y1 ,Y2=Y2,Y3=Y3)



data1 = np.load('final_data.npz' ,allow_pickle=True )
Y1 = data1['Y1']
Y2 = data1['Y2']
Y3 = data1['Y3']
X  = data1['X']
X=X[indexes]

input_img = Input((im_height, im_width, 1), name='img')
n_filters1=10


finding=0
dual_attention_enable_modelss=['vsc ','vs  ','v c ',' sc ', 'v   ','  c ',' s  ','none',]
bb_enable_modelss=['resnet','none']

weight_model_path='weight\\'+'WB3_MSTGANET_model_best3_128_12810_kfold = 9.h5'
path_save='res_GUI\\'
weight_model_path= path_save +'wmodel_MSTGANet_last_' +str(1)+'.h5'                                

cnt_dice=0
from Models_unet import get_model_MSTGANet_model
n_filters1=10
model = get_model_MSTGANet_model (input_img, n_filters = n_filters1, dropout = 0.05, batchnorm = True)
model.load_weights (weight_model_path)

        
p001,p002,p003 = model.predict(X , verbose=1)
index=-1
for i in indexes:
    print(i+1,' from ', indexes,' is done')
    index=index+1

    Y = np.zeros ( [im_width ,  im_height,3])
    Y[:,:,0] = Y1[i,:,:,0]; Y[:,:,1] = Y2[i,:,:,0];Y[:,:,2] = Y3[i,:,:,0]
    p01=p001[index,:,:,0]; p02=p002[index,:,:,0]; p03=p003[index,:,:,0]

    p01=ther(p01, np.mean(p01));p02=ther(p02, np.mean(p02)); p03=ther(p03, np.mean(p03))
    Y[:,:,0]=ther(Y[:,:,0], np.mean(Y[:,:,0]));Y[:,:,1]=ther(Y[:,:,1], np.mean(Y[:,:,1]));Y[:,:,2]=ther(Y[:,:,2], np.mean(Y[:,:,2]))
    p01=smoothing(p01,2);p02=smoothing(p02,2);p03=smoothing(p03,2)
    p01=erosioning(p01,2);p02=erosioning(p02,2);p03=erosioning(p03,2)
    Y[:,:,0]=smoothing(Y[:,:,0],2);Y[:,:,1]=smoothing(Y[:,:,1], 2);Y[:,:,2]=smoothing(Y[:,:,2], 2)
    

    for i in  range(im_width):
        for j in  range(im_height):
            if Y[i,j,2]>=Y[i,j,1] and Y[i,j,2]>=Y[i,j,0] :
                 Y[i,j,1]=0;Y[i,j,0]=0;
            if Y[i,j,1]>=Y[i,j,0] and Y[i,j,1]>=Y[i,j,2] :
                 Y[i,j,0]=0;Y[i,j,2]=0;
            if Y[i,j,0]>=Y[i,j,1] and Y[i,j,0]>=Y[i,j,2] :
                 Y[i,j,1]=0;Y[i,j,2]=0;
                     
    for i in  range(im_width):
        for j in  range(im_height):
            if p03[i,j]>=p02[i,j] and p03[i,j]>=p01[i,j] :
                 p02[i,j]=0;p01[i,j]=0;
            if p02[i,j]>=p01[i,j] and p02[i,j]>=p03[i,j] :
                 p01[i,j]=0;p03[i,j]=0;
            if p01[i,j]>=p02[i,j] and p01[i,j]>=p03[i,j] :
                 p02[i,j]=0;p03[i,j]=0;
    
                 
             
            

    p1p = np.shape(np.where(p01>0))[1]
    p2p = np.shape(np.where(p02>0))[1]
    p3p = np.shape(np.where(p03>0))[1]
    
    if p1p<70: p01=p01*0;# T=T+' 1 ' + str(sumer(p01))
    if p2p<30: p02=p02*0 ;#T=T+' 2 ' + str(sumer(p02))
    if p3p<30: p03=p03*0;#T=T+' 3 ' + str(sumer(p03))

    YY1=np.zeros([ np.shape(p01)[0] ,np.shape(p01)[1],3 ])
    YY1[:,:,0]=p01;    YY1[:,:,1]=p02;    YY1[:,:,2]=p03
    
    d_overal,d_r,d_l,d_b,d_m  = dice_function (Y,YY1)
    print(d_overal , d_r ,d_l,d_b,d_m)
    lenn =6

    d_overal_c= str(d_overal );   d_overal_c =d_overal_c[ 0:min([len(d_overal_c ), lenn ]) ]
    d_r_c= str(d_r );   d_r_c =d_r_c[ 0:min([len(d_r_c ), lenn ]) ]
    d_l_c= str(d_l );   d_l_c =d_l_c[ 0:min([len(d_l_c ), lenn ]) ]
    d_b_c= str(d_b );   d_b_c =d_b_c[ 0:min([len(d_b_c ), lenn ]) ]
    d_m_c= str(d_m );   d_m_c =d_m_c[ 0:min([len(d_m_c ), lenn ]) ]
    
    CMAP='gray'
    
    plt.figure(2000)
    plt.subplot(3,4,1);plt.imshow(Y[:,:,0],cmap=CMAP);plt.ylabel(sumer(Y[:,:,0]));plt.title('Target RV ');plt.grid(False)
    plt.subplot(3,4,2);plt.imshow(p01,cmap=CMAP);plt.title('Dice RV '+str( d_r_c));plt.ylabel(sumer(YY1[:,:,0]));plt.grid(False)
    
    plt.subplot(3,4,5);plt.imshow(Y[:,:,1],cmap=CMAP);plt.ylabel(sumer(Y[:,:,1]));plt.title('Target Myo ');plt.grid(False)
    plt.subplot(3,4,6);plt.imshow(YY1[:,:,1],cmap=CMAP);plt.title('Dice Myo '+str( d_l_c));plt.ylabel(sumer(YY1[:,:,1]));plt.grid(False)
    
    plt.subplot(3,4,9);plt.imshow(Y[:,:,2],cmap=CMAP);plt.ylabel(sumer(Y[:,:,2]));plt.title('Target LV ');plt.grid(False)
    plt.subplot(3,4,10);plt.imshow(YY1[:,:,2],cmap=CMAP);plt.title('Dice LV '+str( d_m_c));plt.ylabel(sumer(YY1[:,:,2]));plt.grid(False)
    plt.subplot(2,2,2);plt.imshow(Y,cmap=CMAP);plt.ylabel(sumer(Y[:,:,2]));plt.title('Target  ');plt.grid(False)
    plt.subplot(2,2,4);plt.imshow(YY1,cmap=CMAP);plt.ylabel(sumer(Y[:,:,2]));plt.title('Predicted  dice = '+str(d_overal_c));plt.grid(False)

    try:os.mkdir('res_dice2')
    except:s=1
    fig = plt.figure(2000)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(1);
    cnt_dice=cnt_dice+1
    fig.savefig('res_dice2\\temp'+str(cnt_dice)+'.png' )
    plt.close(2000)
    plt.close(1000)
    if just_one==1:
        just_one=0
        os.startfile('res_dice2')
    DICE_ALL.append(d_overal)
    plt.figure(5000)
    plt.plot(DICE_ALL)  
    plt.title (' DICE_ALL = ' + str(0.01*int(10000*np.mean(DICE_ALL)) ) )
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(0.5)
    plt.savefig ('all_dice2.png')

        
        
        
