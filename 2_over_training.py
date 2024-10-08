# =============================================================================
# Aghapanah
# =============================================================================
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import metricss
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
# from keras.layers.merge import concatenate, add
from keras.layers import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img    

# import cv2
import time
time0 = time.time() 
# from Models_unet_1 import get_unet
# from Models_unet_1 import get_unet1
from Models_unet import get_unet_3classes
from Models_unet import get_unet_3classes_2
from Models_unet import get_model_MSTGANet_model


# =============================================================================
# Def
# =============================================================================
def mil (a0):
    # a0=123456789
    a1 = int (a0/10**6);    a01=a0-a1*1000000; a2 = int (a01/10**3);    a001=a01-a2*10**3;    a3 = int (a001 )
    A1=str(a1); A2=str(a2);A3=str(a3)
    if not a1==0:
        if a2<100:
            A2='0'+str(a2)
    if not a2==0:
        if a3<100:
            A3='0'+str(a3)
    T =A1+','+A2+','+A3        
    return  T


# =============================================================================
# initialize
# =============================================================================

demo=5
epoch_demo=2
path_save='res_GUI\\'
try:
    os.mkdir(path_save)
except:s=1
os.startfile(path_save)
init_data=0
batch_size0=20
loader=True 
# loader=False
# im_width = 64 ;im_height = 64
th001 = 0.003
im_width = 128;im_height = 128
# im_width = 256;im_height = 256

epochs_no=  300 ;ep0=1
n_filterss =[10]
plt.close("all")
path0=os.getcwd()
data1 = np.load('DATA.npz',allow_pickle=True)
mask = data1['mask']
image = data1['image']
th=0.50
title =['R.V ' , 'Myo' ,'L.V']

color1='bgrkr'
KFOLD = 2
conter=0





MSE_all=[]
for repeation in range( np.shape (n_filterss)[0]):
    n_filters1=n_filterss[repeation]
    file_name ='input3_'+str(im_width)+'_'+str(im_height)+'.npz'
    path_result='result3 '+str(n_filters1)+' '+str(im_width)
    
    print("No. of masks  = ", len(image))
    files  = os.listdir(path0)
    # ss
    X   = np.zeros((len(image), im_height, im_width, 1), dtype=np.float32)
    Y1  = np.zeros((len(image), im_height, im_width, 1), dtype=np.float32)
    Y2  = np.zeros((len(image), im_height, im_width, 1), dtype=np.float32)
    Y3  = np.zeros((len(image), im_height, im_width, 1), dtype=np.float32)
    
    if not file_name in files:
        for index1 in range(len(image) ):
            print(str(index1) +' / '+ str(len(image)))
            img=image[index1];x_img = img_to_array(img)
            x_img = resize(x_img, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
            masks=mask[index1]
            masks = img_to_array(masks)
            i1=np.where (masks==1);i2=np.where (masks==2);i3=np.where (masks==3)
            masks1=0*masks;masks1[i1]=255
            masks2=0*masks;masks2[i2]=255
            masks3=0*masks;masks3[i3]=255
            masks1 = resize(masks1, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
            masks2 = resize(masks2, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
            masks3 = resize(masks3, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
                    
            X[index1] = x_img/255.0; Y1[index1] = masks1/255.0
            Y2[index1] = masks2/255.0; Y3[index1] = masks3/255.0
            
        np.savez(file_name, X = X, Y1=Y1, Y2=Y2, Y3=Y3)
    
    if  file_name in files:
        print('file is existance')
        data1 = np.load(file_name ,allow_pickle=True )
        X = data1['X']
        Y1 = data1['Y1']
        Y2 = data1['Y2']
        Y3 = data1['Y3']
            
        print('file is loaded')
    
    index_data  =np.random.permutation(len(image))
    if demo>0:
        X=X[:demo];Y1=Y1[:demo];
        Y2=Y2[:demo];Y3=Y3[:demo]
    k_no = int (len (index_data)/KFOLD)
    results_kfold_dice_val = []
    
    model_best ='MSTGANET.H5'
    Weight_best ='W'+model_best
    file_name ='input3_'+str(im_width)+'_'+str(im_height)+'.npz'
    
    path_result=  path_save
    conter=conter+1
    model_name_last =path_save+'model_MSTGANet.h5'
    Weight_last=path_save +'wmodel_MSTGANet_last_' +str(conter)+'.h5'
    Weight_model =path_save +'wmodel_MSTGANet_1_' +str(conter)+'_.h5'
    Weight_model2=path_save +'wmodel_reserve_' +str(conter)+'.h5'

    input_img = Input((im_height, im_width, 1), name='img')
    model = get_model_MSTGANet_model (input_img, n_filters = n_filters1, dropout = 0.05, batchnorm = True)
    model.compile(optimizer='adam',loss='mse',  metrics=['accuracy',metricss.dice_coef])
    
    model.save(model_name_last)
    model.save_weights(Weight_model)
    
    a = model.count_params()
    print('Total parameters = ' + mil(a))
    
    callbacks = [
        EarlyStopping(patience=150, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(Weight_model2, verbose=1, save_best_only=True, save_weights_only=True)]
    
    
    acc_all=[]
    loss_all=[]
    loss1=[];loss2=[];loss3=[];
    loss12=[];loss22=[];loss32=[];
    
    acc1=[];acc2=[];acc3=[];
    acc12=[];acc22=[];acc32=[];
    
    dice1=[];dice2=[];dice3=[];
    dice12=[];dice22=[];dice32=[];
    
    
    dice1_m=[];dice2_m=[];dice3_m=[];
    dice12_m=[];dice22_m=[];dice32_m=[];
    
    if demo >0:
        epochs_no  = epoch_demo
        X= X[0:demo] ;# X=X[0:demo];
        Y1=Y1[0:demo];# Y1=Y1[0:demo];
        Y2=Y2[0:demo];# Y22=Y22[0:demo];
        Y3=Y3[0:demo];# Y23=Y23[0:demo];
        
   
    idx =  np.random.permutation(len(X))[0] 
    T = np.zeros([np.shape(Y1)[1] ,np.shape(Y1)[2] ,3])
    T[:,:,0]  =  Y1[idx,:,:,0];
    T[:,:,1]  =  Y2[idx,:,:,0];
    T[:,:,2]  =  Y2[idx,:,:,0];
    
    plt.figure(1 );
    T = np.zeros([np.shape(Y1)[1] ,np.shape(Y1)[2] ,3])
    T[:,:,0]  =  Y1[idx,:,:,0];
    T[:,:,1]  =  Y2[idx,:,:,0];
    T[:,:,2]  =  Y2[idx,:,:,0];
    
    A11=[];A21=[];A12=[];A22=[];A13=[];A23=[];
    B11=[];B21=[];B12=[];B22=[];B13=[];B23=[];
    
    Running_condition =True 
    epoch1=0
    best1=np.inf
    ERROR1=[]
    conter=0
    # =============================================================================
    # START TRAINING
    # =============================================================================
    
    just_one=1;    
    while  Running_condition:
        epoch1=epoch1+1   # for epoch1 in range(epochs_no):
        print('Epoch ' + str(epoch1) +'/' + str(epochs_no))
        
        print( 'Y1 = '+str( np.shape(Y1))) 
        print( 'Y22 = '+str( np.shape(Y2))) 
        print( 'Y23 = '+str( np.shape(Y3))) 
        
# =============================================================================
#             load model weight
# =============================================================================
        if just_one==1:
            just_one=0
            init_weight_model_path='weight\\'+'WB3_MSTGANET_model_best3_128_12810_kfold = 9.h5'
            try:
                # init_weight_model_path= path_save + 'wmodel_MSTGANet_last_1.h5'
                for conter in range(1,100):
                    init_weight_model_path= path_save +'wmodel_MSTGANet_last_' +str(conter)+'.h5'
                    model.load_weights (init_weight_model_path)
                    print(conter,' is loaded')
                
            except:s=1
            
            model.load_weights (init_weight_model_path)
        p1,p2,p3= model.predict(X[:3])

        for i in range (np.shape(p1)[0]):
            PP=p1[i,:,:,0];
            indexx0=np.where(PP<0.5);PP[indexx0]=0;indexx0=np.where(PP>=0.5);PP[indexx0]=1
            p1[i,:,:,0] = PP;    PP=p2[i,:,:,0];
            indexx0=np.where(PP<0.5);PP[indexx0]=0;indexx0=np.where(PP>=0.5);PP[indexx0]=1
            p2[i,:,:,0] = PP;    PP=p3[i,:,:,0];
            indexx0=np.where(PP<0.5);PP[indexx0]=0;indexx0=np.where(PP>=0.5);PP[indexx0]=1
            p3[i,:,:,0] = PP;
        plt.figure(1000);
        
        index=1
        conter=conter+1
        fig_name=path_result+'\\OUT'+str(conter)+' .jpg'
        P=np.zeros((np.shape(p1)[1],np.shape(p1)[2],3))
        P[:,:,0]=p1[index,:,:,0];P[:,:,1]=p2[index,:,:,0];P[:,:,2]=p3[index,:,:,0];
        
        Y=np.zeros((np.shape(Y1)[1],np.shape(Y1)[2],3))
        Y[:,:,0]=Y1[index,:,:,0];Y[:,:,1]=Y2[index,:,:,0];Y[:,:,2]=Y3[index,:,:,0];

        MSE=np.sum(np.abs(Y-P)) 
        D=np.mean(metricss.acc_coef(Y , P))
        plt.subplot(1,2,1);plt.imshow(P);plt.title('Predict MSE =' +str(int(MSE))
                                                   +'/n dice '+str(D));
        plt.grid(False)
        plt.subplot(1,2,2);plt.imshow(Y);plt.title('Target') ;plt.grid(False)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(1);plt.savefig(fig_name);
        plt.close(1000);
        results = model.fit(X, [Y1,Y2,Y3], batch_size=batch_size0, epochs=epoch1, callbacks=callbacks,\
                        validation_data=(X, [Y1,Y2,Y3]))
        model.save_weights(Weight_last)
        print("Saved model to disk")
        MSE_all.append(MSE)
        plt.plot(MSE_all)
        if MSE<=10 or conter>2000:
            Running_condition=False
        
elapsed = time.time() - time0 
H = int (np.fix(elapsed/3600))
M= int (np.fix(( elapsed - H*3600)/60))
S= int (np.fix(( elapsed - H*3600 -M*60)/1))
print ('elapsed time = '+ str(H) + ' : '+str(M) + ' : '+str(S) + '  ')

try:
    from playsound import playsound
    playsound('finish.mp3')
    playsound('finish.wav')
    print('playing sound')
except:
    print('Not playing sound')
