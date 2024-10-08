import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math
import cv2
import sklearn
import tensorflow as tf
from keras.layers import BatchNormalization , Activation, Dropout

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 8]
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D,Concatenate,Add,Dropout,ReLU,Conv2DTranspose,UpSampling2D

keep_scale = 0.2
l1l2 = tf.keras.regularizers.l1_l2(l1=0, l2=0.0005)
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def resblock(x,level='en_l1',filters=64,keep_scale=keep_scale,l1l2=l1l2,downsample=False,bn_act=True,first_layer=False):
    if downsample:
        if not first_layer:
            x_H = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_Hconv')(x)
            x = Conv2D(filters/2,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
        else:
            x_H = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_Hconv')(x)
            x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
    else:
        x_H = x
        x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
    x = BatchNormalization(name=level+'_conv1bn')(x)
    x = ReLU(name=level+'_conv1relu')(x)
    if downsample:
        x = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_conv2')(x)
    else:
        x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv2')(x)
    x = BatchNormalization(name=level+'_conv2bn')(x)
    x = ReLU(name=level+'_conv2relu')(x)
    x = Add(name=level+'_add')([x_H*keep_scale,x*(1-keep_scale)])
    if bn_act:
        x = BatchNormalization(name=level+'_finalbn')(x)
        x = ReLU(name=level+'_finalrelu')(x)       
    return x

# def get_model(image_size=256):
#   inputs = tf.keras.Input(shape=(image_size,image_size,3),name='input')
def get_model(input_img,  n_filters = 32, dropout = 0.1, batchnorm = True):

  inputs = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
  
  layer_0 = Conv2D(64,(7,7),strides=(2,2),padding='same',name='input_conv1',activation=None)(inputs)
  layer_0 = tf.image.resize(layer_0, [112,112], method=tf.image.ResizeMethod.BILINEAR)
  layer = BatchNormalization(name='input_conv1bn')(layer_0)
  layer = ReLU(name='input_conv1relu')(layer)

  ## Encoder
  layer = resblock(layer,'en_l1',64,keep_scale,l1l2,downsample=True,bn_act=True,first_layer=True)
  layer = resblock(layer,'en_l2',64,keep_scale,l1l2,downsample=False,bn_act=True)
  encoder_1 = resblock(layer,'en_l3',64,keep_scale,l1l2,downsample=False,bn_act=False)
  encoder_1 = tf.image.resize(encoder_1, [56, 56], method=tf.image.ResizeMethod.BILINEAR)
  layer = BatchNormalization(name='en_l3_finalbn')(encoder_1)
  layer = ReLU(name='en_l3_finalrelu')(layer)

  layer = resblock(layer,'en_l4',128,keep_scale,l1l2,downsample=True,bn_act=True)
  layer = resblock(layer,'en_l5',128,keep_scale,l1l2,downsample=False,bn_act=True)
  layer = resblock(layer,'en_l6',128,keep_scale,l1l2,downsample=False,bn_act=True)
  encoder_2 = resblock(layer,'en_l7',128,keep_scale,l1l2,downsample=False,bn_act=False)
  encoder_2 = tf.image.resize(encoder_2, [28, 28], method=tf.image.ResizeMethod.BILINEAR)
  layer = BatchNormalization(name='en_l7_finalbn')(encoder_2)
  layer = ReLU(name='en_l7_finalrelu')(layer)

  layer = resblock(layer,'en_l8',256,keep_scale,l1l2,downsample=True,bn_act=True)
  layer = resblock(layer,'en_l9',256,keep_scale,l1l2,downsample=False,bn_act=True)
  layer = resblock(layer,'en_l10',256,keep_scale,l1l2,downsample=False,bn_act=True)
  layer = resblock(layer,'en_l11',256,keep_scale,l1l2,downsample=False,bn_act=True)
  layer = resblock(layer,'en_l12',256,keep_scale,l1l2,downsample=False,bn_act=True)
  encoder_3 = resblock(layer,'en_l13',256,keep_scale,l1l2,downsample=False,bn_act=False)
  layer = BatchNormalization(name='en_l13_finalbn')(encoder_3)
  layer = ReLU(name='en_l13_finalrelu')(layer)

  layer = resblock(layer,'en_l14',512,keep_scale,l1l2,downsample=True,bn_act=True)
  layer = resblock(layer,'en_l15',512,keep_scale,l1l2,downsample=False,bn_act=True)
  layer = resblock(layer,'en_l16',512,keep_scale,l1l2,downsample=False,bn_act=True)

  ## DAC block
  b1 = Conv2D(512,(3,3),padding='same',dilation_rate=1,name='dac_b1_conv1',activation=None)(layer)
  # b1 = BatchNormalization()(b1)
  # b1 = ReLU(name='dac_b1_relu')(b1)

  b2 = Conv2D(512,(3,3),padding='same',dilation_rate=3,name='dac_b2_conv1',activation=None)(layer)
  b2 = Conv2D(512,(1,1),padding='same',dilation_rate=1,name='dac_b2_conv2',activation=None)(b2)
  # b2 = BatchNormalization()(b2)
  # b2 = ReLU(name='dac_b2_relu')(b2)

  b3 = Conv2D(512,(3,3),padding='same',dilation_rate=1,name='dac_b3_conv1',activation=None)(layer)
  b3 = Conv2D(512,(3,3),padding='same',dilation_rate=3,name='dac_b3_conv2',activation=None)(b3)
  b3 = Conv2D(512,(1,1),padding='same',dilation_rate=1,name='dac_b3_conv3',activation=None)(b3)
  # b3 = BatchNormalization()(b3)
  # b3 = ReLU(name='dac_b3_relu')(b3)

  b4 = Conv2D(512,(3,3),padding='same',dilation_rate=1,name='dac_b4_conv1',activation=None)(layer)
  b4 = Conv2D(512,(3,3),padding='same',dilation_rate=3,name='dac_b4_conv2',activation=None)(b4)
  b4 = Conv2D(512,(3,3),padding='same',dilation_rate=5,name='dac_b4_conv3',activation=None)(b4)
  b4 = Conv2D(512,(1,1),padding='same',dilation_rate=1,name='dac_b4_conv4',activation=None)(b4)
  # b4 = BatchNormalization()(b4)
  # b4 = ReLU(name='dac_b4_relu')(b4)

  layer = Add(name='dac_add')([layer,b1,b2,b3,b4])
  # layer = BatchNormalization(name='dac_bn')(layer)
  layer = ReLU(name='dac_relu')(layer)

  ## RMP block
  b1 = MaxPool2D((2,2),strides=(2,2),padding='valid',name='rmp_b1_pool')(layer)
  b1 = Conv2D(1,(1,1),padding='valid',name='rmb_b1_conv1',activation=None)(b1)
  b1 = Conv2DTranspose(1,(1,1),(2,2),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b1)
  b1 = tf.image.resize(b1, [14,14], method=tf.image.ResizeMethod.BILINEAR)

  b2 = MaxPool2D((3,3),strides=(3,3),padding='valid',name='rmp_b2_pool')(layer)
  b2 = Conv2D(1,(1,1),padding='valid',name='rmb_b2_conv1',activation=None)(b2)
  b2 = Conv2DTranspose(1,(1,1),(3,3),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b2)
  b2 = tf.image.resize(b2, [14,14], method=tf.image.ResizeMethod.BILINEAR)

  b3 = MaxPool2D((5,5),strides=(5,5),padding='valid',name='rmp_b3_pool')(layer)
  b3 = Conv2D(1,(1,1),padding='valid',name='rmb_b3_conv1',activation=None)(b3)
  b3 = Conv2DTranspose(1,(1,1),(5,5),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b3)
  b3 = tf.image.resize(b3, [14,14], method=tf.image.ResizeMethod.BILINEAR)

  b4 = MaxPool2D((6,6),strides=(6,6),padding='valid',name='rmp_b4_pool')(layer)
  b4 = Conv2D(1,(1,1),padding='valid',name='rmb_b4_conv1',activation=None)(b4)
  b4 = Conv2DTranspose(1,(1,1),(6,6),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b4)
  b4 = tf.image.resize(b4, [14,14], method=tf.image.ResizeMethod.BILINEAR)
  layer = tf.image.resize(layer, [14,14], method=tf.image.ResizeMethod.BILINEAR)
  layer = Concatenate(name='rmp_concat')([layer,b1,b2,b3,b4])

  layer = ReLU(name='rmp_relu')(layer)

  layer = Conv2D(256,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l1_conv1',activation=None)(layer)
  layer = BatchNormalization(name='de_l1_conv1bn')(layer)
  layer = ReLU(name='de_l1_conv1relu')(layer)
  layer = Conv2DTranspose(256,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l1_deconv2',activation=None)(layer)
  layer = BatchNormalization(name='de_l1_conv2bn')(layer)
  layer = ReLU(name='de_l1_deconv2relu')(layer)
  layer = Conv2D(256,(3,3),padding='same',kernel_regularizer=l1l2,name='de_l1_conv3',activation=None)(layer)
  layer = tf.image.resize(layer, [14,14], method=tf.image.ResizeMethod.BILINEAR)
  layer = BatchNormalization(name='de_l1_conv3bn')(layer)
  layer = ReLU(name='de_l1_conv3relu')(layer)

  layer = Conv2D(128,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l2_conv1',activation=None)(layer)
  layer = BatchNormalization(name='de_l2_conv1bn')(layer)
  layer = ReLU(name='de_l2_conv1relu')(layer)
  layer = Conv2DTranspose(128,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l2_deconv2',activation=None)(layer)
  layer = BatchNormalization(name='de_l2_conv2bn')(layer)
  layer = ReLU(name='de_l2_deconv2relu')(layer)
  layer = Conv2D(128,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l2_conv3',activation=None)(layer)
  layer = Add(name='de_l2_add')([encoder_2*keep_scale,layer*(1-keep_scale)])
  layer = BatchNormalization(name='de_l2_conv3bn')(layer)
  layer = ReLU(name='de_l2_conv3relu')(layer)

  layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l3_conv1',activation=None)(layer)
  layer = BatchNormalization(name='de_l3_conv1bn')(layer)
  layer = ReLU(name='de_l3_conv1relu')(layer)
  layer = Conv2DTranspose(64,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l3_deconv2',activation=None)(layer)
  layer = BatchNormalization(name='de_l3_conv2bn')(layer)
  layer = ReLU(name='de_l3_deconv2relu')(layer)
  layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l3_conv3',activation=None)(layer)
  layer = Add(name='de_l3_add')([encoder_1*keep_scale,layer*(1-keep_scale)])
  layer = BatchNormalization(name='de_l3_conv3bn')(layer)
  layer = ReLU(name='de_l3_conv3relu')(layer)

  layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l4_conv1',activation=None)(layer)
  layer = BatchNormalization(name='de_l4_conv1bn')(layer)
  layer = ReLU(name='de_l4_conv1relu')(layer)
  layer = Conv2DTranspose(64,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l4_deconv2',activation=None)(layer)
  layer = BatchNormalization(name='de_l4_conv2bn')(layer)
  layer = ReLU(name='de_l4_deconv2relu')(layer)
  layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l4_conv3',activation=None)(layer)
  layer = Add(name='de_l4_add')([layer_0*keep_scale,layer*(1-keep_scale)])
  layer = BatchNormalization(name='de_l4_conv3bn')(layer)
  layer = ReLU(name='de_l4_conv3relu')(layer)

  layer = Conv2DTranspose(32,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name='final_deconv1',activation=None)(layer)
  layer = BatchNormalization()(layer)
  layer = ReLU()(layer)
  layer = Conv2D(32,(3,3),padding='same',kernel_regularizer=l1l2,name='final_conv1',activation=None)(layer)
  layer = BatchNormalization()(layer)
  layer = ReLU()(layer)
  layer = tf.image.resize(layer, [128, 128], method=tf.image.ResizeMethod.BILINEAR)
  
  
  
  outputs = Conv2D(4,(3,3),padding='same',name='output',activation='softmax')(layer)

  
  out_title =['Right_Ventricle' , 'Myocard' ,'Left_Ventricle']
  o1 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[0])(layer)
  o2 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[1])(layer)
  o3 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[2])(layer)
  #1 is number of class
  model = tf.keras.Model(inputs=[input_img], outputs=[o1,o2,o3],name='CE-Net')
    
  # model = tf.keras.Model(inputs,outputs,name='CE-Net')
  return model
# model = get_model()
# model.summary()