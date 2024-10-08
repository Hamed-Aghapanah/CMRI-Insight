"""   Created on Fri May 26 11:12:08 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""
# =============================================================================
# USER Guide
# =============================================================================

# in this file we interduce VIT and dual attention (position and channel)
# in segmentation of CMRI

# Notice : We use 3 output consist of 3 masks 'Right_Ventricle' , 'Myocard' ,'Left_Ventricle'
# and other data are assume as back ground

# load Libs: is loading lib in numerical and deep learning

# warrning off : is for switch GPU warning to off

# Initialize  to Create : for Initialization train and test  data 
# modes: are modes for Net such as  dual ,MSTGANET,NNUNET,CENET
# path_models='my models vit2' save folder

# path_name_save_every_epoch1 = '2_SBBVDAMLF_'
# path_name_save_every_epoch='path_name_save_every_epoch1'+name_back_bone+'_'+dual_attention_enable_model_N+'_loss '+str(loss_N)
# out_title : labels of 3 mask  ['Right_Ventricle' , 'Myocard' ,'Left_Ventricle']


# function : our functions: 
# color_mask
# find_index_good to find a good data for showing 
# weight_loss : sum errors
# saver_result : save coloerd result
# show_and_save_coeff_exel in each epoch we save result in exel file 
#   Our ViT DUAL attention model
# class VIT_function(keras.layers.Layer): is base on ShiftedPatchTokenization
# model's functions to create our function on Vannil Unet
# def upsample_conv2d(img):
# def upsample(img):
# def downsample(img):
# def downsample2(img):
# init_layer(layer):
# blocks
# conv2d_block
# conv_block_simple
# conv_block_simple_no_bn
# identity_block
# BB_Resnet
# back_bone
# Encoder_Block0
# Encoder_Block
# Decoder_Block
# Decoder_Block0
# Decoder_Block1
# conv2d_block2
# VDAB_block
# DAB2in
# MSDAB
# VDAB_block
# DAB2in
# MSDAB
# BBVDAMLF_model


# see VIT model
# show and save all models  if show_all_model is enable
# show model Dual attention if show_one_model is enable  

#   Training ablation study
# ablation_study
# we try                             print('FINISH 4 training stage 1 ')
# we try                             print('FINISH 7 training stage 2 ')
# we try                     Fine Tuning


#                   important saving 
# name='final_model mode_'+mode+
# ' bb_'+ name_back_bone+
# ' DA_'+dual_attention_enable_model_N+' loss_N_'+str(loss_N)


  
# =============================================================================
# load Libs
# =============================================================================
# from models_RASTI_NN_Unet import get_model as get_model_NN_Unet
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
from  saed import *
import keras as K
from skimage.transform import resize
import os
import time
time0 = time.time() 
from metricss import *
# =============================================================================
# warrning off
# =============================================================================
import warnings
import sys
# import shutup; shutup.please()
if not sys.warnoptions:
    warnings.simplefilter("ignore")
def fxn():warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings("ignore")
warnings.warn('my warning')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")
plt.close('all')
# =============================================================================
# Initialize  to Create 
# =============================================================================
# modes=['dual']
# modes=['MSTGANET']
# modes=['NNUNET']
# modes=['CENET']

demo=False
demo=True

show_print=True
# show_print=False

path_models='my models vit2'
path_name_save_every_epoch1 = '2_SBBVDAMLF_'
img_size_target = 128

# detail_size_attention=True
detail_size_attention=False

show_one_model=False
# show_one_model=True

# porposed_model=True
# porposed_model=False

play_sound = True
play_sound = False

show_save_result_every_epochs=False
show_save_result_every_epochs =True

ablation_study=True
# ablation_study=False

# show_all_model=True
show_all_model=False

preprocessing=False
 


shutdown_after_Train_model=True
shutdown_after_Train_model=False

apply_our_losses=True
apply_our_losses=False

ceof_loss_callibrate=True
# ceof_loss_callibrate=False
global alfas4
global alfas7
global out_title
global loss_N
global loss_loss_loss
global loss1_cof1
global loss1_cof2
global loss1_cof3
global loss1_cof4
global loss1_cof5
global loss1_cof6
global loss1_cof7
global index001_train
global index001_val
global index001_test



out_title =['Right_Ventricle' , 'Myocard' ,'Left_Ventricle']
# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
# metrics_all =['accuracy' ,'dice',tf.keras.metrics.MeanIoU(num_classes=2),tf.keras.metrics.MeanAbsoluteError,tf.keras.metrics.Precision,tf.keras.metrics.Recall,tf.keras.metrics.TrueNegatives,tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()]

# metrics_all =['accuracy' ,'dice',tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.BinaryIoU(num_classes=2),\
#               tf.keras.metrics.MeanAbsoluteError,tf.keras.metrics.Precision,tf.keras.metrics.Recall,tf.keras.metrics.TrueNegatives,tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()]

    
metrics_all =['accuracy' , dice_coef]
# modes=['CENET','NNUNET','dual','MSTGANET']
modes=['dual','MSTGANET','CENET','NNUNET']
# modes=[ 'MSTGANET' ,'dual']
dual_attention_enable_modelss=['vsc ','vs  ','v c ',' sc ', 'v   ','  c ',' s  ','none',]
# dual_attention_enable_modelss=['vsc  ']
loss_no=[1,4,7]
# loss_no=[7]
# loss_no=[1]

bb_enable_modelss=['resnet','none']
# bb_enable_modelss=['resnet' ]

epsilon = 10**-5

EPOCH7 = 1000
EPOCH4 = EPOCH7+200 
EPOCH1 = copy.deepcopy(EPOCH4)
EPOCH_fine_tuning = 200
EPOCH_CAL=1


EPOCH7 = 200
EPOCH4 = EPOCH7+200 
EPOCH1 = copy.deepcopy(EPOCH4)
EPOCH1=220
EPOCH_fine_tuning = 200
EPOCH_CAL=1

if demo:
    EPOCH7 = 3
    EPOCH4 = EPOCH7+2 
    EPOCH1 = copy.deepcopy(220)
    EPOCH_fine_tuning = 2
    EPOCH_CAL=1

INIT_LR = 1e-3
batch_size_no = 2
metricsss = [ 'accuracy',
             # metricss.iou
    ]
 
# =============================================================================
# function
# =============================================================================
def color_mask(a,layer=1):
    color_mask1=np.zeros( [np.shape(a)[0],np.shape(a)[1] ,3])
    color_mask1[:,:,layer-1]=a[:,:,0]
    return color_mask1   
    

    
def find_index_good(Y_train10,Y_train20,Y_train30,th=0.1):
    solid=[]
    index001_train=0
    for index001_train in range(len(Y_train10)):
        tempp =np.mean(Y_train10[index001_train]) *np.mean(Y_train30[index001_train]) *np.mean(Y_train30[index001_train]) 
        solid.append(tempp)
        # if np.mean(Y_train10[index001_train]) >th and  np.mean(Y_train20[index001_train]) >th and   np.mean(Y_train30[index001_train]) >th :
        #     break
        # # print(index001_train)
    
    try:index001_train=np.where (solid ==np.max(solid))
    except :s=1
    
    try: index001_train=index001_train[0]
    except :s=1
    
    try: index001_train=index001_train[0]
    except :s=1
    
    try: index001_train=index001_train[0]
    except :s=1
    
    print('index001 = ' , index001_train)
     
    return index001_train    
        
def weight_loss (y_true, y_pred):
    try:
        import metricss
        if loss_N ==4:   
            error=0
            x1='l1=loss1_cof1 *' +loss_loss_loss[0] +'(y_true, y_pred)'
            # print('123456798',x1)
            # eval (x1)
            eval ('l2=loss1_cof2 *' +loss_loss_loss[1] +'(y_true, y_pred)')
            eval ('l3=loss1_cof3 *' +loss_loss_loss[2] +'(y_true, y_pred)')
            eval ('l4=loss1_cof4 *' +loss_loss_loss[3] +'(y_true, y_pred)')
            eval ('l5=loss1_cof5 *' +loss_loss_loss[4] +'(y_true, y_pred)')
            # eval ('l6=loss1_cof6 *' +loss_loss_loss[5] +'(y_true, y_pred)')
            # eval ('l7=loss1_cof7 *' +loss_loss_loss[6] +'(y_true, y_pred)')
            error = l1+l2+l3+l4+l5
              
        if loss_N ==7:
            error=0
            eval ('l1=loss1_cof1 *' +loss_loss_loss[0] +'(y_true, y_pred)')
            eval ('l2=loss1_cof2 *' +loss_loss_loss[1] +'(y_true, y_pred)')
            eval ('l3=loss1_cof3 *' +loss_loss_loss[2] +'(y_true, y_pred)')
            eval ('l4=loss1_cof4 *' +loss_loss_loss[3] +'(y_true, y_pred)')
            eval ('l5=loss1_cof5 *' +loss_loss_loss[4] +'(y_true, y_pred)')
            eval ('l6=loss1_cof6 *' +loss_loss_loss[5] +'(y_true, y_pred)')
            eval ('l7=loss1_cof7 *' +loss_loss_loss[6] +'(y_true, y_pred)')
            error = l1+l2+l3+l4+l5+l6+l7
            
    except:
        from metricss import losss
        error = losss(y_true, y_pred)
    return error


def saver_result (model,X,Y1,Y2,Y3,index,path_name_save_every_epoch,current_epoch,
                  train_val_test='train'):
    import matplotlib.pyplot as plt
    import os
    # index=4
    # dicee=0.5
    # print('ssss',X.shape)
    p1,p2,p3 = model.predict(X , verbose=1)
    
    # plt.figure();
    XX,YY    =  masker (X,index,Y1,Y2,Y3)
    XX1,YY1  =  masker (X,index,p1,p2,p3)

    dicee1 = np.mean (metricss.acc_coef(Y1[index], p1[index]))
    dicee2 = np.mean (metricss.acc_coef(Y2[index], p2[index]))
    dicee3 = np.mean (metricss.acc_coef(Y3[index], p3[index]))
    
    diff1 = np.mean (np.power(Y1[index]- p1[index],2))
    diff2 = np.mean (np.power(Y2[index]- p2[index],2))
    diff3 = np.mean (np.power(Y3[index]- p3[index],2))
    
    dicee1=100*np.round(dicee1,4)
    dicee2=100*np.round(dicee2,4)
    dicee3=100*np.round(dicee3,4)
    
    diff1=100*np.round(diff1,4)
    diff2=100*np.round(diff2,4)
    diff3=100*np.round(diff3,4)
    
    
    dicee=(dicee3+dicee2+dicee1)/3
    difff=(diff3+diff2+diff1)/3
    dicees=np.round(dicee,4)
    diffs=np.round(difff,4)
    plt.figure()
    plt.subplot(141);plt.imshow(X[index]);plt.title('Original Image , index = '+str(index))  
    plt.subplot(142);plt.imshow(YY);plt.title('Ground Truth')  
    plt.subplot(143);plt.imshow(YY1);plt.title('Predicted '+str(dicees)+' %')  
    plt.subplot(143);plt.imshow(YY1);plt.title('Predicted MSE = '+str(diffs) )  
    plt.subplot(144);plt.imshow((YY-YY1));plt.title('Diffrence ')  
    try:os.mkdir(path_name_save_every_epoch)
    except:print('hast')
    
    os.startfile(path_name_save_every_epoch)
    
    try:
        # plt.legend();
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(5);
        plt.savefig ( path_name_save_every_epoch +'\\'+train_val_test+' F_index_'+str(index)+'ep = '+str(current_epoch)+'.png')
        plt.close('all')
    except:print('hast')
    plt.figure()
    
    dicee1s =  np.round(dicee1,4)
    dicee2s =  np.round(dicee2,4)
    dicee3s =  np.round(dicee3,4)
    
    diff1s =  np.round(diff1,4)
    diff2s =  np.round(diff2,4)
    diff3s =  np.round(diff3,4)
    
    cnt=0
    plt.subplot(1,2,1);plt.imshow(X[index],cmap='gray');plt.title('Original Image , index = '+str(index))  
    
    cnt=4    ;plt.subplot(3,6,cnt);plt.imshow(Y1[index],cmap='gray');plt.title('Ground Truth '+out_title[0])  
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(p1[index],cmap='gray');
    
    temp1=color_mask  (Y1[index] , 1)
    temp2=color_mask  (p1[index] , 1)
    cnt=4    ;plt.subplot(3,6,cnt);plt.imshow(temp1,cmap='gray');plt.title('Ground Truth '+out_title[0])  
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(temp2,cmap='gray');
    
    
    plt.title('Predicted '+ str(dicee1s)+' %' )
    plt.title('Predicted MSE = '+ str(diff1s)  )
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(np.abs (Y1[index]-p1[index]),cmap='gray');plt.title('Diffrence ')  
       
    cnt=10   ;plt.subplot(3,6,cnt);plt.imshow(Y2[index],cmap='gray');plt.title('Ground Truth '+out_title[1])  
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(p2[index],cmap='gray');
    
    temp1=color_mask  (Y2[index] , 2)
    temp2=color_mask  (p2[index] , 2)
    cnt=10    ;plt.subplot(3,6,cnt);plt.imshow(temp1,cmap='gray');plt.title('Ground Truth '+out_title[0])  
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(temp2,cmap='gray');
    
    plt.title('Predicted '+ str(dicee2s)+' %' )
    plt.title('Predicted MSE = '+ str(diff2s) )
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(np.abs (Y2[index]-p2[index]),cmap='gray');plt.title('Diffrence ')  
    
    cnt=16   ;plt.subplot(3,6,cnt);plt.imshow(Y3[index],cmap='gray');plt.title('Ground Truth '+out_title[2])  
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(p3[index],cmap='gray');
    
    temp1=color_mask  (Y3[index] , 3)
    temp2=color_mask  (p3[index] , 3)
    cnt=16    ;plt.subplot(3,6,cnt);plt.imshow(temp1,cmap='gray');plt.title('Ground Truth '+out_title[0])  
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(temp2,cmap='gray');
    
     
    plt.title('Predicted '+ str(dicee3s)+' %' )
    plt.title('Predicted MSE = '+ str(diff3s) )
    cnt=cnt+1;plt.subplot(3,6,cnt);plt.imshow(np.abs (Y3[index]-p3[index]),cmap='gray');plt.title('Diffrence ')  
    
    
    plt.subplot(1,2,1);plt.imshow(X[index],cmap='gray');plt.title ('Original Image , index = '+str(index))  
    
    # plt.subplot(144);plt.imshow(YY-YY1);plt.title('Diffrence ')  
    
    try:os.mkdir(path_name_save_every_epoch)
    except:print('hast')
    
    try:

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(5);
         
        plt.savefig ( path_name_save_every_epoch +'\\'+train_val_test+' index_'+str(index)+'ep = '+str(current_epoch)+'.png')
        plt.close('all')
    except:print('hast')
    s=1
    
def show_and_save_coeff_exel ( epoch, model,history , file_exel ,show=True, save=True):
    print (10*'*')
    print ('show_and_save_coeff_exel')
    print (10*'*')
    final_metrics=[]
    final_value=[]
    try:
        keyss=history.history.keys()
        for key1 in  keyss:
            # if 'loss' in key1  or 'accuracy' in key1:
            final_metrics.append(key1)
        if show_print:
            print('final_metrics',final_metrics)
        for S in final_metrics:
            
            v=history.history[S][-1]
            final_value.append(v)
    except:s=1
    metrics1 =   final_metrics  
    mvalue=final_value
    sheet_name='model_parameters'
    counter0=0;lenx=[];values=[];model_layers_names=[];indexes=[]
    for i in range(np.shape(model.layers) [0]): 
        # print(i,model.layers[i].name)
        if  '_channel' in model.layers[i].name or '_position' in model.layers[i].name\
            or '_VIT' in model.layers[i].name or '_original' in model.layers[i].name :
            x=model.layers[i].name
            xx = str(counter0)+' i='+str(i)+' layers.name = '+str(x) 
            lenx.append (8+len (xx))
    for i in range(np.shape(model.layers) [0]):        
        if  '_channel' in model.layers[i].name or '_position' in model.layers[i].name\
            or '_VIT' in model.layers[i].name or '_original' in model.layers[i].name :
            counter0=counter0+1
            
            x=model.layers[i].name
            v=tf.nn.softplus(model.layers[i].w).numpy()[0]
            temp=''
            if v>0:
                temp=' '
            t=''
            if i<1000:t=' '
            if i<100:t='  '
            if i<10:t='   '
            t1=''
            if counter0<1000:t1=' '
            if counter0<100:t1='  '
            if counter0<10:t1='   '
            
            xx = str(counter0)+t1+' i='+str(i)+t+' layers.name = '+str(x) 
            if show:
                if show_print:
                    print (xx,(np.max(lenx)-len (xx))*' ','value = ',temp,v)
            values.append(v)
            model_layers_names.append(x)
            indexes.append(counter0)
    C=[];    V=[]
    
    V.append(epoch)
    C.append('epoch')
    if save:
        for m in mvalue:
            V.append(m)
        V.append('#')
        for m in values:
            V.append(m)
        
    
        for m in metrics1:
            C.append(m)
        C.append('#')    
        for m in model_layers_names:
            C.append(m)
        
        df = pd.DataFrame([V], columns=C,)# index=indexes, 
        with pd.ExcelWriter(file_exel) as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    return 1 



# =============================================================================
# Keras costume layer
# =============================================================================

class coef_layer(keras.layers.Layer):
    def __init__(self, name,**kwargs):
        super(coef_layer, self).__init__(name=name)
        w_init = tf.random_normal_initializer()
        # w_init =tf.keras.constraints.NonNeg()
        self.w = tf.Variable(
            # initial_value =tf.keras.constraints.NonNeg(),
            initial_value=w_init(shape=(1,), dtype="float32"),
            trainable=True,
        )
        # self.name = name
        #self.w = tf.nn.softplus(self.w2)
        
    def call(self, inputs):
        # tf.math.greater_equal(w,0.0)
        # w.tf.cast(tf.gra
        # self.add_loss(keras.constraints.non_neg(self.w))
        # ln (e^x +1)
        #self.w = tf.nn.softplus(self.w2)
        return inputs *  tf.nn.softplus(self.w)



# =============================================================================
#   Our ViT DUAL attention model
# =============================================================================
# ShiftedPatchTokenization
class VIT_function(keras.layers.Layer):
    def __init__(
        self,
        image_size=(None,128,128,1),
        patch_size=32,
        num_patches=128,
        projection_dim=1280,
        vanilla=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = keras.layers.Reshape((num_patches, -1))
        self.projection = keras.layers.Dense(units=projection_dim)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        })
        return config
    
    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        # return (tokens, patches)
        tokens = keras.layers.Reshape(self.image_size[1:])(tokens)
        tokens = tf.keras.activations.softmax(tokens)
# 
        
        return  tokens 
# myVit = VIT_function(image_size=(None,128,128,1),
# patch_size=32,
# num_patches=128,
# projection_dim=1280,
# vanilla=True,)
# print(myVit)

# stop_here_now
# =============================================================================
# end VIT
# =============================================================================


# =============================================================================
# model's functions 
# =============================================================================
def upsample_conv2d(img):
    if img_size_ori == img_size_target:
        return img
    return Conv2DTranspose(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
   

    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    #return img[:img_size_ori, :img_size_ori]

def downsample2(img):
    x = MaxPooling2D((2, 2))(img)
    return x


# conv strid =2 

def init_layer(layer):
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)



# =============================================================================
# blocks
# =============================================================================
def conv2d_block(input_tensor, n_filters=10, kernel_size = 3, batchnorm = True):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv



def identity_block(input_tensor, kernel_size=3, filters=[3,3,3], stage='stage1', block='b1'):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
    

def BB_Resnet(w,h):
    
    try:
        base_model = ResNet50(include_top=False , input_shape=(128,128,3), pooling='avg', weights=None)
    except:    
        
        # Transfer learning 
        # base_model=Keras.models.load ('ResNet50')
        # base_model=Keras.models.load_weights('wresnet50')
        s=1
    # base_model = ResNet50(include_top=False, weights=None , input_shape=(w,h,3), pooling='avg')
    """
    for i in range(143):
        if base_model.layers[i].output.shape[1] == 64 and  'conv' in base_model.layers[i].name[-4:]:
            i64=i;print(i, base_model.layers[i].name)
            break
    for i in range(143):        
        if base_model.layers[i].output.shape[1] == 32 and  'conv' in base_model.layers[i].name[-4:]:
            i32=i;print(i, base_model.layers[i].name)
            break
    for i in range(143):        
        if base_model.layers[i].output.shape[1] == 16 and  'conv' in base_model.layers[i].name[-4:]:
            i16=i;print(i, base_model.layers[i].name)
            break
    for i in range(143):        
        if base_model.layers[i].output.shape[1] == 8 and  'conv' in base_model.layers[i].name[-4:]:
            i8=i;print(i, base_model.layers[i].name)
            break
            
    """        
            
    resnet_base = keras.models.Model(base_model.input, base_model.layers[142].output)
    # from keras.models import Model
    input_shape=(w,h,3)
    x=(w,h,3)
    # Build model.
    #model = Model(input_shape, x, name='resnet50')

    # resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    for l in resnet_base.layers:
        l.trainable = True
    #conv1 = resnet_base.get_layer("input_7").output #  ==> 128
    conv1 = resnet_base.layers[0].output
    #conv2 = resnet_base.get_layer("conv1_relu").output   # Layer 4==> 64
    conv2 = resnet_base.layers[4].output
    conv3 = resnet_base.layers[38].output
    conv4 = resnet_base.layers[80].output
    conv5 = resnet_base.layers[142].output
    
    # conv3 = resnet_base.get_layer("conv2_block3_out").output  # Layer 38  ==> 32
    # conv4 = resnet_base.get_layer("conv3_block4_out").output  # Layer 80  ==> 16
    # conv5 = resnet_base.get_layer("conv4_block6_out").output # Layer 142  ==> 8

    up6 = concatenate(axis=-1)([UpSampling2D()(conv5), conv4])
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate(axis=-1)([UpSampling2D()(conv6), conv3] )
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate(axis=-1)([UpSampling2D()(conv7), conv2])
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate(axis=-1)([UpSampling2D()(conv8), conv1])
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    # up10 = UpSampling2D()(conv9)
    conv10 = conv_block_simple(conv9, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model 

def back_bone(input_tensor,name ='resnet', BBtrainable=False):
    # name_back_bone ='mobilenet_v2'
    # name_back_bone ='vgg16'
    # name_back_bone ='inception_resnet_v2

    import numpy as np
    # x = conv2d_block(input_tensor)
    S = np.shape(input_tensor)
    # print( 'input size = ',S)
    im_height=S[1];    im_width=S[2]
    from tensorflow.keras.applications.resnet50 import ResNet50
    BB_model = BB_Resnet(S[0],S[1])
    
    BB_model.trainable = BBtrainable
    # print('BBmodel output: ', len(BB_model.layers), BB_model.output.shape)
    
    # if 'resnet50' in name.lower():
    #     from tensorflow.keras.applications.resnet50 import ResNet50
    #     BB_model = ResNet50(include_top=False,input_shape=(im_height,im_width,1),weights=None,pooling='avg')
    #     print( 'ResNet50 is loaded')
    # if 'mobile' in name.lower():
    #     from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    #     BB_model = MobileNetV2(include_top=False,input_shape=(im_height,im_width,1),weights=None,pooling='avg')
    #     print( 'MobileNetV2 is loaded')
        
    # if 'vgg' in name.lower():
    #     from tensorflow.keras.applications.vgg16 import VGG16
    #     BB_model = VGG16(include_top=False,input_shape=(im_height,im_width,1),weights=None,pooling='avg')
    #     print( 'VGG16 is loaded')
        
    # if 'inception' in name.lower():
    #     from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    #     BB_model = InceptionResNetV2(include_top=False,input_shape=(im_height,im_width,1),weights=None,pooling='avg')
    #     print( 'InceptionResNetV2 is loaded')
        
        
    f001 =BB_model (input_tensor)
    # print (np.shape(f001))
    a = BB_model.count_params()
    # print('Total Parameters  (Back BONE) = ',name, mil(a))
     
# =============================================================================
#     shape and name  layer
# =============================================================================
    return f001

def Encoder_Block0(input_tensor, n_filters =10, kernel_size = 3, batchnorm = True,
                   dual_attention_enable_Encoder_Block0='sc',
                   section_name_Encoder_Block0 = 'section_name_Encoder_Block0'
                   
                   ):
    x = conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    x =VDAB_block(x ,3,
                    kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_Encoder_Block0,
                                    section_name = section_name_Encoder_Block0
                                    )
    
    return x

# Encoder_Block
# MAXPool ,Conv2DB ,Conv2DB
def Encoder_Block(input_tensor, n_filters=10, kernel_size = 3, batchnorm = True,
                  dual_attention_enable_Encoder_Block='sc',
                  section_name_Encoder_Block = 'section_name_Encoder_Block'
                  ):
    x = MaxPooling2D((2, 2))(input_tensor)
    x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x =VDAB_block(x ,3,
                    kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_Encoder_Block ,
                                    section_name = section_name_Encoder_Block
                                    )
    return x
# Conv2DB ,Conv2DB

# Decoder_Block
# Conv2DB , Conv2DB Up-Sample, Conv2DB
def Decoder_Block(A2, B3, n_filters=10, kernel_size = 3, batchnorm = True,
                dual_attention_enable_Decoder_Block='vsc',
                section_name_Decoder_Block = 'section_name_Decoder_Block'
                   ):
    
    
    input_tensor = concatenate(axis=-1)([A2, B3] )
    
    # x = MaxPooling2D((2, 2))(input_tensor)
    x = conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x=  UpSampling2D( size=(2, 2),  interpolation="nearest")(x)
    x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x =VDAB_block(x ,3,
                    kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_Decoder_Block ,
                                    section_name = section_name_Decoder_Block
                                    )
    return x
# Decoder_Block0
# Conv1 ×1, Sigmoid
def Decoder_Block0(A1,B2 ,  n_filters=10, kernel_size = 3, batchnorm = True,
                   dual_attention_enable_Decoder_Block0='vsc',
                   section_name_Decoder_Block0 = 'section_name_Decoder_Block0'
                   ):
    x = concatenate(axis=-1)([A1, B2])
    
    D1 =VDAB_block(x ,n_filters,
                    kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_Decoder_Block0 ,
                                    section_name = section_name_Decoder_Block0
                                    )
    
    
    # x = conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x = Conv2D(1, (1, 1), activation='sigmoid')(D1)
    
    # o1 = Conv2D(1, (1, 1), activation='sigmoid',name= 'R_V_')(x)
    # o2 = Conv2D(1, (1, 1), activation='sigmoid',name= 'MyoCard')(x)
    # o3 = Conv2D(1, (1, 1), activation='sigmoid',name= 'L_V_')(x)
    
    o1 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[0])(x)
    o2 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[1])(x)
    o3 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[2])(x)
    
    # out_title
    
    return o1,o2,o3
# Backward Block1
# Up-Sample, Conv2DB
def Decoder_Block1(input_tensor, n_filters=10, kernel_size = 3, batchnorm = True
                    ,
                    
                    dual_attention_enable_Decoder_Block1='vsc',
                    section_name_Decoder_Block1 = 'section_name_Decoder_Block1'
                    ):
    # x = MaxPooling2D((2, 2))(input_tensor)
    # x = conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    # x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    x=  UpSampling2D( size=(2, 2),  interpolation="nearest")(input_tensor)
    x = conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    x =VDAB_block(x ,n_filters,
                    kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_Decoder_Block1 ,
                                    section_name = section_name_Decoder_Block1
                                    )
    
    return x

def conv2d_block2(input_tensor, n_filters=10, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def VDAB_block(input_tensor, n_filters=10, kernel_size = 3, batchnorm = True,
                dual_attention_enable='vsc',
                section_name = 'section name'):
    x_att_cross=True
    # print('##################################################')
    # print(input_tensor)
    # print('input_tensor.shape',input_tensor.shape)
    
    
    input_tensor = downsample2 (input_tensor)
    
    if detail_size_attention:
        print(10*'=',section_name)
        print(10*'I','input_tensor=',input_tensor.shape)
    # input_tensor = input_tensor[:,::2,::2,:]
    
    ATTENTION=0
    
    att_show=0
    if 'c' in dual_attention_enable:
        x_CAM=CAM()(input_tensor)
        beta_x_att_ch = coef_layer (name=section_name + '_channel')(x_CAM)
        ATTENTION=ATTENTION+beta_x_att_ch
        att_show=att_show+1
        if detail_size_attention:
            print(10*'C','CAM=',beta_x_att_ch.shape)
            print(10*'A','ATTENTION=',ATTENTION.shape)
    if 's' in dual_attention_enable:
        x_PAM=PAM()(input_tensor)
        alpha_x_att_pos = coef_layer (name=section_name + '_position')(x_PAM)
        ATTENTION=ATTENTION+alpha_x_att_pos
        att_show=att_show+10
        if detail_size_attention:
            print(10*'S','PAM=',alpha_x_att_pos.shape)
            print(10*'A','ATTENTION=',ATTENTION.shape)
    if 'v' in dual_attention_enable:
        x_VIT = VIT_function(image_size=input_tensor.shape,
        patch_size=32,
        num_patches=input_tensor.shape[1],
        projection_dim=input_tensor.shape[2]*input_tensor.shape[3],
        vanilla=True)(input_tensor)
        
        # print('sss', input_tensor.shape)
        # if input_tensor.shape[1]==64: load weight 
        #     ff
        #     h5 load
        #     dic
        #     key
        #     value
        
        
        gamma_x_att_vit = coef_layer (name=section_name + '_VIT')(x_VIT)
        ATTENTION=ATTENTION+gamma_x_att_vit
        att_show=att_show+100
        
        if detail_size_attention:
            print(10*'V','VIT=',gamma_x_att_vit.shape)
            print(10*'A','ATTENTION=',ATTENTION.shape)
    if att_show>0:
        out_att_block       =  input_tensor  +  ATTENTION
        if x_att_cross:
            out_att_block   = input_tensor  * ATTENTION
    
    if att_show==0:
        out_att_block=input_tensor
    # print('x.shape = ', x.shape)
    out_att_block = keras.layers.UpSampling2D((2,2))( out_att_block )
        
    
    return out_att_block

def DAB2in(Di, Ej ,n_f ,
                kernel_size = 3, batchnorm = True,
                dual_attention_enable_DAB2in='vsc',
                section_name_DAB2in = 'section name DAB2in' ):
    DiEj = concatenate(axis=-1)([Di, Ej])
    
    Ai =VDAB_block(DiEj ,n_f,
                    kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_DAB2in ,
                                    section_name = section_name_DAB2in
                                    )
    return Ai

def MSDAB(E4, E3, E2,E1 ,n_f ,dual_attention_enable_MSDAB='vsc',section_name_MSDAB = 'section_MSDAB'):
    
    E42=  AveragePooling2D( pool_size=(1, 1),  strides=None, padding="valid", data_format=None)(E4)
    E32=  AveragePooling2D( pool_size=(2, 2),  strides=None, padding="valid", data_format=None)(E3)
    E22=  AveragePooling2D( pool_size=(4, 4),  strides=None, padding="valid", data_format=None)(E2)
    E12=  AveragePooling2D( pool_size=(8, 8),  strides=None, padding="valid", data_format=None)(E1)
    
    
    DiEj = concatenate(axis=-1)([E42, E32, E22,E12])
    Ai =VDAB_block(DiEj ,n_f
                    , kernel_size = 3, batchnorm = True,
                                    dual_attention_enable=dual_attention_enable_MSDAB,
                                    section_name = section_name_MSDAB
                                    
                                    )
    return Ai

def BBVDAMLF_model(input_img,
                   bb_name='resnet', 
                   dual_attention_enable_model='vsc',
                   n_filters = 10, dropout = 0.1, batchnorm = True):   
    # print(bb_name ,' is selected as back bone')
    # print(dual_attention_enable_model ,' is selected as dual attention ')
    # print(multi_loss_no ,' is selected as multi loss no')

    name=bb_name
    
    if not 'none' in name:
        #input_img = concatenate(-1)([input_img]*3)
        _,w,h,c = input_img.shape
        
        
        # print(w,h,c)
        
        # PATCH  PATCH
        # F0_1 = back_bone (input_img[:,:int(w//2),:int(h//2)],name+'_1' )
        # F0_2 = back_bone (input_img[:, int(w//2):,:int(h//2)],name+'_2' )
        # F0_3 = back_bone (input_img[:,:int(w//2),int(h//2):],name+'_3' )
        # F0_4 = back_bone (input_img[:, int(w//2):, int(h//2):],name+'_4' )
        # F0h1 = concatenate(axis=1)([F0_1, F0_2])
        # F0h2 = concatenate(axis=1)([F0_3, F0_4])
        # F0 = concatenate(axis=2)([F0h1, F0h2])


        
        # input_img = input_img[:,::2,::2,:]
        F0 = back_bone (input_img,name)
        # F0 = keras.layers.UpSampling2D((2,2))(F0)
        
        
        # print('t'*20)
        # print(F0)
        
        E1 = Encoder_Block0 (F0,dual_attention_enable_Encoder_Block0=dual_attention_enable_model,
        section_name_Encoder_Block0 = 'section_Encoder1')
    if 'none' in name:
        E1 = Encoder_Block0 (input_img,dual_attention_enable_Encoder_Block0=dual_attention_enable_model,
        section_name_Encoder_Block0 = 'section_Encoder1')
        
    E2 = Encoder_Block (E1,dual_attention_enable_Encoder_Block=dual_attention_enable_model,
    section_name_Encoder_Block = 'section_Encoder2')
    
    
    E3 = Encoder_Block (E2,dual_attention_enable_Encoder_Block=dual_attention_enable_model,
    section_name_Encoder_Block = 'section_Encoder3')
    
    E4 = Encoder_Block (E3,dual_attention_enable_Encoder_Block=dual_attention_enable_model,
    section_name_Encoder_Block = 'section_Encoder4')
    
    A4 = MSDAB(E4, E3, E2,E1 ,  n_filters * 8,dual_attention_enable_MSDAB=dual_attention_enable_model)
    E4 =Decoder_Block1(A4 ,
                        dual_attention_enable_Decoder_Block1=dual_attention_enable_model,
                        section_name_Decoder_Block1 = 'section_Decoder4'
                        )
    # print('.'*60)
    # print('E3, B4',E3.shape, B4.shape)

    # print('.'*60)
    # print('E1, E2', E1.shape, E2.shape)

    A3 = DAB2in(E3, E4 ,  n_filters * 8,
                kernel_size = 3, batchnorm = True,
                                                dual_attention_enable_DAB2in=dual_attention_enable_model,
                                                section_name_DAB2in = 'section_name_A3')
    
    E3 =Decoder_Block(A3,E4 ,
                       dual_attention_enable_Decoder_Block=dual_attention_enable_model,
                       section_name_Decoder_Block = 'section_Decoder3'
                       )
    # print('.'*60)
    # print('E2,B3', E2.shape, B3.shape)
    A2 = DAB2in(E2, E3 ,  n_filters * 8,
                kernel_size = 3, batchnorm = True,
                                                dual_attention_enable_DAB2in=dual_attention_enable_model,
                                                section_name_DAB2in = 'section_name_A2')
    # print('.'*60)
    # print('A2', A2.shape)

    E2 =Decoder_Block(A2,E3,
                       dual_attention_enable_Decoder_Block=dual_attention_enable_model,
                       section_name_Decoder_Block = 'section_Decoder2')
    A1 = DAB2in(E1, E2 ,  n_filters * 8,
                kernel_size = 3, batchnorm = True,
                                                dual_attention_enable_DAB2in=dual_attention_enable_model,
                                                section_name_DAB2in = 'section_name_A1')
    o1,o2,o3 =Decoder_Block0(A1,E2,
                             dual_attention_enable_Decoder_Block0=dual_attention_enable_model,
                             section_name_Decoder_Block0 = 'section_Decoder1'
                             )
    
    
    model = Model(inputs=[input_img], outputs=[o1,o2,o3])
    return model



