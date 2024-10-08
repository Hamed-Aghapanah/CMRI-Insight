import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, concatenate, BatchNormalization, LeakyReLU, ZeroPadding2D, Input
from tensorflow.keras import Model
import numpy as np
from skimage.transform import resize
from keras.layers.convolutional import Conv2D, Conv2DTranspose

def nnUNet_2D(image_shape, feature_maps=32, max_fa=480, num_pool=8, k_init='he_normal', n_classes=4):
                          
    x = Input(image_shape)
    inputs = x
    # x = resize(x, (256, 256, 1), mode = 'constant', preserve_range = True)     
    # x = Conv2D(filters = 2, kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same' )(x)                                                                                                  
    x= tf.keras.layers.Resizing(256, 256, interpolation="bilinear", crop_to_aspect_ratio=False)(x)

    
        
    l = []
    seg_outputs = []
    fa_save = []
    fa = feature_maps

    # ENCODER
    x = StackedConvLayers(x, fa, k_init, first_conv_stride=1)
    fa_save.append(fa)
    fa = fa*2 if fa*2 < max_fa else max_fa
    l.append(x)

    # conv_blocks_context
    for i in range(num_pool-1):
        x = StackedConvLayers(x, fa, k_init)
        fa_save.append(fa)
        fa = fa*2 if fa*2 < max_fa else max_fa
        l.append(x)

    # BOTTLENECK
    x = StackedConvLayers(x, fa, k_init, first_conv_stride=(1,2))

    # DECODER
    for i in range(len(fa_save)):
        # tu
        if i == 0:
            x = Conv2DTranspose(fa_save[-(i+1)], (1, 2), use_bias=False,
                                strides=(1, 2), padding='valid') (x)
        else:
            x = Conv2DTranspose(fa_save[-(i+1)], (2, 2), use_bias=False,
                                strides=(2, 2), padding='valid') (x)
        x = concatenate([x, l[-(i+1)]])

        # conv_blocks_localization
        x = StackedConvLayers(x, fa_save[-(i+1)], k_init, first_conv_stride=1)
        seg_outputs.append(Conv2D(n_classes, (1, 1), use_bias=False, activation="softmax")(x))   
    
    outputs = seg_outputs[-1]  
    
    outputs= tf.keras.layers.Resizing(128, 128, interpolation="bilinear", crop_to_aspect_ratio=False)(outputs)


    # print('np.shape(outputs) =' , np.shape(outputs))
    out_title =['Right_Ventricle' , 'Myocard' ,'Left_Ventricle']
    o1 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[0])(outputs)
    o2 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[1])(outputs)
    o3 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[2])(outputs)
    #1 is number of class
    model = tf.keras.Model(inputs=[inputs], outputs=[o1,o2,o3],name='NN-Net')
      
    # model = Model(inputs=[inputs], outputs=[outputs])

    # Calculate the weigts as nnUNet does
    ################# Here we wrap the loss for deep supervision ############
    # we need to know the number of outputs of the network
    net_numpool = num_pool

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    weights = weights[::-1] 
    ################# END ###################

    return model

def StackedConvLayers(x, feature_maps, k_init, first_conv_stride=2):
    x = ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=first_conv_stride)
    x = ConvDropoutNormNonlin(x, feature_maps, k_init)
    return x

    
def ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=1):
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(feature_maps, (3, 3), strides=first_conv_stride, activation=None,
               kernel_initializer=k_init, padding='valid') (x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1) (x)
    x = LeakyReLU(alpha=0.01) (x)
    return x


# model = nnUNet_2D(image_shape = (256,256,1))
# model = nnUNet_2D(image_shape = (128,128,1))
# model.save('s.h5')
# import os
# os.startfile('s.h5')

# model.summary()