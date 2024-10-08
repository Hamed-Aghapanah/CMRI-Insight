
import numpy as np
# =============================================================================
# def
# =============================================================================
def Grad_CAM1 (model,X,Y1,Y2,Y3,index=0,tavan=1, tavan2=1 , smoothed=True,alpha=0.4,alpha2=0.6):
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

    # from tensorflow.keras.applications.
    import numpy as np
    import pandas as pd
    # from  saed import *
    import keras as K
    from skimage.transform import resize
    import os
    import time
    time0 = time.time() 
    # from metricss import *
    try:
        Y = np.zeros ( [np.shape(X[index])[0] ,  np.shape(X[index])[1],3])
        Y[:,:,0] = Y1[index,:,:,0]
        Y[:,:,1] = Y2[index,:,:,0]
        Y[:,:,2] = Y3[index,:,:,0]
    except:
        Y = np.zeros ( [np.shape(X[index])[0] ,  np.shape(X[index])[1],3])
        Y[:,:,0] = Y1[index,:,:]
        Y[:,:,1] = Y2[index,:,:]
        Y[:,:,2] = Y3[index,:,:]
        
    index_train=range(index+1)
    index_train=[index,index]
    X=X[index_train];
    e=0.00001
    cnt=0
    # X=X[index]
    for layer in model.layers:
        cnt=cnt+1
    
    p01,p02,p03 = model.predict(X , verbose=1)
    # plt.figure()
    # plt.subplot(131);plt.imshow(p01[0,:,:,0],cmap='gray')
    # plt.subplot(132);plt.imshow(p02[0,:,:,0],cmap='gray')
    # plt.subplot(133);plt.imshow(p03[0,:,:,0],cmap='gray')
    # index=2
    # print(np.shape())
    XX1,YY1  =  masker (X,0,p01,p02,p03)
    plt.figure()
    plt.subplot(2,3,1);plt.imshow(X[0],cmap='gray' );plt.title('Original image')
    plt.subplot(2,3,2);plt.imshow(Y,cmap='gray' );plt.title('Original Masks')
    print('np.max(Y)',np.max(Y))
    print('np.max(p01)',np.max(p01))
    
    print('np.shape(Y)',np.shape(Y))
    print('np.shape(p01)',np.shape(p01))
    
    
    
    # d_r = np.mean (metricss.acc_coef(Y[:,:,0], p01[0,:,:,0]))
    # d_l = np.mean (metricss.acc_coef(Y[:,:,1], p02[0,:,:,0]))
    # d_m = np.mean (metricss.acc_coef(Y[:,:,2], p03[0,:,:,0]))
    Y[:,:,0]=(Y[:,:,0] -np.min(Y[:,:,0]))/(np.max(Y[:,:,0]) -np.min(Y[:,:,0]+0.001 ))
    Y[:,:,1]=(Y[:,:,1] -np.min(Y[:,:,1]))/(np.max(Y[:,:,1]) -np.min(Y[:,:,1]+0.001 ))
    Y[:,:,2]=(Y[:,:,2] -np.min(Y[:,:,2]))/(np.max(Y[:,:,2]) -np.min(Y[:,:,2]+0.001 ))

    # p01[0,:,:,0]=(p01[0,:,:,0] -np.min(p01[0,:,:,0]))/(np.max(p01[0,:,:,0]) -np.min(p01[0,:,:,0]+0.001 ))
    # p01[0,:,:,0]=(p01[0,:,:,0] -np.min(p01[0,:,:,0]))/(np.max(p01[0,:,:,0]) -np.min(p01[0,:,:,0]+0.001 ))
    # p02[0,:,:,0]=(p02[0,:,:,0] -np.min(p02[0,:,:,0]))/(np.max(p02[0,:,:,0]) -np.min(p02[0,:,:,0]+0.001 ))

    d_r =roundd (2*np.sum(Y[:,:,0]*p01[0,:,:,0])/np.sum((Y[:,:,0]+p01[0,:,:,0])))
    d_l =roundd (2*np.sum(Y[:,:,1]*p01[0,:,:,0])/np.sum((Y[:,:,1]+p01[0,:,:,0])))
    d_m =roundd (2*np.sum(Y[:,:,2]*p01[0,:,:,0])/np.sum((Y[:,:,2]+p01[0,:,:,0])))
    d_overal= (d_r+d_l+d_m)/3
    
    
    ddd=str(d_r) +'_'+str(d_l) +'_'+str(d_m) +'_'+str(d_overal) 
    plt.subplot(2,3,3);plt.imshow( YY1 ,cmap='gray' );plt.title('Predicted Masks \n'+'dice ='+ddd)
    
    # xxxx
    last_conv_layer_name1 =model.layers[-1].name
    last_conv_layer_name2 =model.layers[-2].name
    last_conv_layer_name3 =model.layers[-3].name
    # for i in range(5):
    #     model.layers[-1*i-1].activation = None

    model.layers[-1].activation = None
    model.layers[-2].activation = None
    model.layers[-3].activation = None
    
    grad_model1 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name1).output])
    grad_model2 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name2).output])
    grad_model3 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name3).output])
    grad_model=tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name1).output ,model.get_layer(last_conv_layer_name2).output,model.get_layer(last_conv_layer_name3).output])
    
    p1   = grad_model1.predict(X , verbose=1)
    p2   = grad_model2.predict(X , verbose=1)
    p3   = grad_model3.predict(X , verbose=1)
    
    # plt.figure()
    # plt.subplot(131);plt.imshow(p1[0,:,:,0],cmap='gray')
    # plt.subplot(132);plt.imshow(p2[0,:,:,0],cmap='gray')
    # plt.subplot(133);plt.imshow(p3[0,:,:,0],cmap='gray')
    
    # p    = grad_model.predict(X , verbose=1)
    # p1=p[0]
    # p2=p[1]
    # p3=p[2]
    # plt.close('all')
    p11=  p1 [0] ;p22=  p2 [0] ;p33=  p3 [0] 
    
    p11=128*(p11-np.min(p11)-e)/ (np.max(p11)-np.min(p11)-e);
    p22=128*(p22-np.min(p22)-e)/ (np.max(p22)-np.min(p22)-e);
    p33=160*(p33-np.min(p33)-e)/ (np.max(p33)-np.min(p33)-e)
    p1=p1[:,:,0];p2=p2[:,:,0];p3=p3[:,:,0]
    
    p11=p11**tavan;p22=p22**tavan;p33=p33**tavan
    p11=p11**1/tavan;p22=p22**1/tavan;p33=p33**1/tavan
    
    p=p11+p22+p33
    if smoothed:
        import cv2
        for i in range(3):
            p = cv2.medianBlur(p, 5)
    # plt.figure();
    # plt.imshow( p ,cmap='jet' )
    p=np.uint8( (255)*(p-np.min(p)-e)/ (np.max(p)-np.min(p)-e) );
    # plt.figure();
    plt.subplot(2,3,4);    plt.imshow(  p ,cmap='jet' );plt.title('Grad-CAM '+'(Withput Activation Function)')
    X1 = X[0]
    X3 = np.zeros ( [np.shape(X1)[0] ,  np.shape(X1)[1],3])
    X3[:,:,0]=X1[:,:,0];X3[:,:,1]=X1[:,:,0];X3[:,:,2]=X1[:,:,0]
    # plt.figure();plt.imshow( X3 ,cmap='gray' )
    pp = cv2.cvtColor(p,cv2.COLOR_GRAY2RGB)
    heatmap = cv2.cvtColor(pp,cv2.COLOR_RGB2GRAY)
    
    import matplotlib.cm as cm

    img=X3
    heatmap = np.uint8(  heatmap)# Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")# Use RGB values of the colormap
    
    
    # plt.close('all')
    jet = cm.get_cmap("gist_rainbow")
    jet = cm.get_cmap("rainbow")
    # jet = cm.get_cmap("gist_ncar")
    # jet = cm.get_cmap("bone")
    # jet = cm.get_cmap("hot")
    # jet = cm.get_cmap("gray")
    jet = cm.get_cmap("jet")
    
    
    jetgray = cm.get_cmap("gray")
    jetgray_colors = jetgray(np.arange(256))[:, :3]
    jetgray_heatmap = jetgray_colors[heatmap]
    jetgray_heatmap=jetgray_heatmap**tavan2
    
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap=jet_heatmap**tavan2
    
    
   
    jet_heatmap2=1*(jet_heatmap-np.min(jet_heatmap)-e)/ (np.max(jet_heatmap)-np.min(jet_heatmap)-e);
    superimposed_img1 = jet_heatmap2 * alpha + img* alpha2 # superimposed_img2 = (jet_heatmap2 * 0.6)* (img)
    plt.subplot(2,3,5);plt.imshow(superimposed_img1 ) # plt.figure();plt.imshow(superimposed_img2 ) 
    plt.title(' Superimposed Visualization')
    jet_heatmap=jet_heatmap*jetgray_heatmap
    # plt.figure();plt.imshow(jet_heatmap)# Create an image with RGB colorized heatmap
    jet_heatmap2=1*(jet_heatmap-np.min(jet_heatmap)-e)/ (np.max(jet_heatmap)-np.min(jet_heatmap)-e);
    superimposed_img = jet_heatmap2 * alpha + img* alpha2 # superimposed_img2 = (jet_heatmap2 * 0.6)* (img)
    plt.subplot(2,3,6);plt.imshow(superimposed_img ) # plt.figure();plt.imshow(superimposed_img2 ) 
    plt.title(' Superimposed Visualization2')
    
    return YY1 , p,superimposed_img1, superimposed_img
    

def Grad_CAM2 (model,X,Y1,Y2,Y3,index=0,tavan=1, tavan2=1 , smoothed=True,alpha=0.4,alpha2=0.6):
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

    # from tensorflow.keras.applications.
    import numpy as np
    import pandas as pd
    # from  saed import *
    import keras as K
    from skimage.transform import resize
    import os
    import time
    time0 = time.time() 
    # from metricss import *
    index=1
    Y = np.zeros ( [np.shape(X[index])[0] ,  np.shape(X[index])[1],3])
    Y[:,:,0] = Y1[index,:,:,0]
    Y[:,:,1] = Y2[index,:,:,0]
    Y[:,:,2] = Y3[index,:,:,0]
    
    index_train=range(index+1)
    index_train=[index,index]
    X=X[index_train];
    e=0.00001
    cnt=0
    # X=X[index]
    for layer in model.layers:
        cnt=cnt+1
    
    p01,p02,p03 = model.predict(X , verbose=1)
    # plt.figure()
    # plt.subplot(131);plt.imshow(p01[0,:,:,0],cmap='gray')
    # plt.subplot(132);plt.imshow(p02[0,:,:,0],cmap='gray')
    # plt.subplot(133);plt.imshow(p03[0,:,:,0],cmap='gray')
    # index=2
    # print(np.shape())
    XX1,YY1  =  masker (X,0,p01,p02,p03)
    plt.figure()
    plt.subplot(2,3,1);plt.imshow(X[0],cmap='gray' );plt.title('Original image')
    plt.subplot(2,3,2);plt.imshow(Y,cmap='gray' );plt.title('Original Masks')
    print('np.max(Y)',np.max(Y))
    print('np.max(p01)',np.max(p01))
    
    print('np.shape(Y)',np.shape(Y))
    print('np.shape(p01)',np.shape(p01))
    
    
    
    # d_r = np.mean (metricss.acc_coef(Y[:,:,0], p01[0,:,:,0]))
    # d_l = np.mean (metricss.acc_coef(Y[:,:,1], p02[0,:,:,0]))
    # d_m = np.mean (metricss.acc_coef(Y[:,:,2], p03[0,:,:,0]))
    Y[:,:,0]=(Y[:,:,0] -np.min(Y[:,:,0]))/(np.max(Y[:,:,0]) -np.min(Y[:,:,0]+0.001 ))
    Y[:,:,1]=(Y[:,:,1] -np.min(Y[:,:,1]))/(np.max(Y[:,:,1]) -np.min(Y[:,:,1]+0.001 ))
    Y[:,:,2]=(Y[:,:,2] -np.min(Y[:,:,2]))/(np.max(Y[:,:,2]) -np.min(Y[:,:,2]+0.001 ))

    # p01[0,:,:,0]=(p01[0,:,:,0] -np.min(p01[0,:,:,0]))/(np.max(p01[0,:,:,0]) -np.min(p01[0,:,:,0]+0.001 ))
    # p01[0,:,:,0]=(p01[0,:,:,0] -np.min(p01[0,:,:,0]))/(np.max(p01[0,:,:,0]) -np.min(p01[0,:,:,0]+0.001 ))
    # p02[0,:,:,0]=(p02[0,:,:,0] -np.min(p02[0,:,:,0]))/(np.max(p02[0,:,:,0]) -np.min(p02[0,:,:,0]+0.001 ))

    d_r =roundd (2*np.sum(Y[:,:,0]*p01[0,:,:,0])/np.sum((Y[:,:,0]+p01[0,:,:,0])))
    d_l =roundd (2*np.sum(Y[:,:,1]*p01[0,:,:,0])/np.sum((Y[:,:,1]+p01[0,:,:,0])))
    d_m =roundd (2*np.sum(Y[:,:,2]*p01[0,:,:,0])/np.sum((Y[:,:,2]+p01[0,:,:,0])))
    d_overal= (d_r+d_l+d_m)/3
    
    
    ddd=str(d_r) +'_'+str(d_l) +'_'+str(d_m) +'_'+str(d_overal) 
    plt.subplot(2,3,3);plt.imshow( YY1 ,cmap='gray' );plt.title('Predicted Masks \n'+'dice ='+ddd)
    
    # xxxx
    last_conv_layer_name1 =model.layers[-1].name
    last_conv_layer_name2 =model.layers[-2].name
    last_conv_layer_name3 =model.layers[-3].name
    # for i in range(5):
    #     model.layers[-1*i-1].activation = None

    model.layers[-1].activation = None
    model.layers[-2].activation = None
    model.layers[-3].activation = None
    
    grad_model1 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name1).output])
    grad_model2 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name2).output])
    grad_model3 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name3).output])
    grad_model=tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name1).output ,model.get_layer(last_conv_layer_name2).output,model.get_layer(last_conv_layer_name3).output])
    
    p1   = grad_model1.predict(X , verbose=1)
    p2   = grad_model2.predict(X , verbose=1)
    p3   = grad_model3.predict(X , verbose=1)
    
    # plt.figure()
    # plt.subplot(131);plt.imshow(p1[0,:,:,0],cmap='gray')
    # plt.subplot(132);plt.imshow(p2[0,:,:,0],cmap='gray')
    # plt.subplot(133);plt.imshow(p3[0,:,:,0],cmap='gray')
    
    # p    = grad_model.predict(X , verbose=1)
    # p1=p[0]
    # p2=p[1]
    # p3=p[2]
    # plt.close('all')
    p11=  p1 [0] ;p22=  p2 [0] ;p33=  p3 [0] 
    
    p11=128*(p11-np.min(p11)-e)/ (np.max(p11)-np.min(p11)-e);
    p22=128*(p22-np.min(p22)-e)/ (np.max(p22)-np.min(p22)-e);
    p33=160*(p33-np.min(p33)-e)/ (np.max(p33)-np.min(p33)-e)
    p1=p1[:,:,0];p2=p2[:,:,0];p3=p3[:,:,0]
    tavan=1
    tavan2=1
    smoothed=True
    alpha=0.4
    alpha2=0.6
    
    p11=p11**tavan;p22=p22**tavan;p33=p33**tavan
    p11=p11**1/tavan;p22=p22**1/tavan;p33=p33**1/tavan
    
    p=p11+p22+p33
    if smoothed:
        import cv2
        for i in range(3):
            p = cv2.medianBlur(p, 5)
    # plt.figure();
    # plt.imshow( p ,cmap='jet' )
    p=np.uint8( (255)*(p-np.min(p)-e)/ (np.max(p)-np.min(p)-e) );
    # plt.figure();
    plt.subplot(2,3,4);    plt.imshow(  p ,cmap='jet' );plt.title('Grad-CAM '+'(Withput Activation Function)')
    X1 = X[0]
    X3 = np.zeros ( [np.shape(X1)[0] ,  np.shape(X1)[1],3])
    X3[:,:,0]=X1[:,:,0];X3[:,:,1]=X1[:,:,0];X3[:,:,2]=X1[:,:,0]
    # plt.figure();plt.imshow( X3 ,cmap='gray' )
    pp = cv2.cvtColor(p,cv2.COLOR_GRAY2RGB)
    heatmap = cv2.cvtColor(pp,cv2.COLOR_RGB2GRAY)
    
    import matplotlib.cm as cm

    img=X3
    heatmap = np.uint8(  heatmap)# Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")# Use RGB values of the colormap
    
    
    # plt.close('all')
    jet = cm.get_cmap("gist_rainbow")
    jet = cm.get_cmap("rainbow")
    # jet = cm.get_cmap("gist_ncar")
    # jet = cm.get_cmap("bone")
    # jet = cm.get_cmap("hot")
    # jet = cm.get_cmap("gray")
    jet = cm.get_cmap("jet")
    
    
    jetgray = cm.get_cmap("gray")
    jetgray_colors = jetgray(np.arange(256))[:, :3]
    jetgray_heatmap = jetgray_colors[heatmap]
    jetgray_heatmap=jetgray_heatmap**tavan2
    
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap=jet_heatmap**tavan2
    
    
   
    jet_heatmap2=1*(jet_heatmap-np.min(jet_heatmap)-e)/ (np.max(jet_heatmap)-np.min(jet_heatmap)-e);
    superimposed_img1 = jet_heatmap2 * alpha + img* alpha2 # superimposed_img2 = (jet_heatmap2 * 0.6)* (img)
    plt.subplot(2,3,5);plt.imshow(superimposed_img1 ) # plt.figure();plt.imshow(superimposed_img2 ) 
    plt.title(' Superimposed Visualization')
    jet_heatmap=jet_heatmap*jetgray_heatmap
    # plt.figure();plt.imshow(jet_heatmap)# Create an image with RGB colorized heatmap
    jet_heatmap2=1*(jet_heatmap-np.min(jet_heatmap)-e)/ (np.max(jet_heatmap)-np.min(jet_heatmap)-e);
    superimposed_img = jet_heatmap2 * alpha + img* alpha2 # superimposed_img2 = (jet_heatmap2 * 0.6)* (img)
    plt.subplot(2,3,6);plt.imshow(superimposed_img ) # plt.figure();plt.imshow(superimposed_img2 ) 
    plt.title(' Superimposed Visualization2')
    
    return YY1 , p,superimposed_img1, superimposed_img
    

def MSEE(a,b):
    M=1
    len_a=np.sum(0*a+1)
    # print(len_a)
    if np.max(a)>1 or  np.max(b)>1 :
        M=255**2;
    s=np.sum ((a-b)**2 ) /(M*len_a)
    return s


def get_divisors(n):
    s=[]
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            s.append(i)
    return s



def mil (a0):
    a1  =  int (a0/10**6);    a01 = a0-a1*10**6; 
    a2  =  int (a01/10**3);    a001 = a01-a2*10**3;
    a3  =  int (a001 )
    A1 = str(a1)+','; A2 = str(a2)+',';A3 = str(a3)
    if a1==0:A1=''
    if not a1==0:
        if a2<100:A2 = '0'+str(a2)+','
        if a2<10:A2 = '00'+str(a2)+','
    if not a2==0:
        if a3<100:A3 = '0'+str(a3)
        if a3<10:A3 = '00'+str(a3)
    
    
    T  = A1+A2+A3        
    if a1==0 and a2==0:T=A3
    
    for i in range(12-len (T)):
        T=' '+T
    
    return  T 

def roundd(a=123.456,b=1):  # a=0.123456789;   # b=3
    aa = a*100*10**b;
    aa= int (aa);
    aa=aa/(10**b)
    return aa

def masker (X,index,Y1,Y2,Y3) :
    index=0
    print(np.shape (X),np.shape (Y1),np.shape (Y2),np.shape (Y3),)
    X1=X[index]
    X3 = np.zeros ( [np.shape(X1)[0] ,  np.shape(X1)[1],3])
    X3[:,:,0]=X1 [:,:,0];X3[:,:,1]=X1 [:,:,0];X3[:,:,2]=X1 [:,:,0]
    YY1=0*X3;
    try:
        YY1[:,:,0]=Y1 [index,:,:,0];
        YY1[:,:,1]=Y2 [index,:,:,0];
        YY1[:,:,2]=Y3 [index,:,:,0]
    except:
        YY1[:,:,0]=Y1 [index,:,: ];
        YY1[:,:,1]=Y2 [index,:,: ];
        YY1[:,:,2]=Y3 [index,:,: ]
    
    XX=X3
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(XX)[1]):
            for k in range(np.shape(XX)[2]):
                if YY1[i,j,k]>0:     XX[i,j,:]=0;XX[i,j,k]=1
    return XX,YY1


def standardize(train, test):
        import numpy as np
        mean = np.mean(train, axis=0)
        std = np.std(train, axis=0)+0.000001
        X_train = (train - mean) / std
        X_test = (test - mean) /std
        return X_train, X_test
    