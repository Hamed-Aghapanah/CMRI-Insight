from keras.models import Model

from keras.layers import BatchNormalization , Activation, Dropout

# from keras.layers import  BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate as  concatenate

# from tensorflow.keras.layers.merge import concatenate
# from keras.layers import Concatenate as  concatenate
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # # second layer
    # x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
    #           kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    # if batchnorm:
    #     x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    
    return x

def conv2d_block2(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
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

def get_unet_3classes(input_img,number_class=3, n_filters = 32, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    # p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    # p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    # p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    # p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    
    
    u6_1 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6_1 = concatenate(axis=-1)([u6_1, c4])
    # u6 = Dropout(dropout)(u6)
    c6_1 = conv2d_block(u6_1, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    u7_1 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6_1)
    u7_1 = concatenate(axis=-1)([u7_1, c3])
    # u7 = Dropout(dropout)(u7)
    c7_1 = conv2d_block(u7_1, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u8_1 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7_1)
    u8_1 = concatenate(axis=-1)([u8_1, c2])
    # u8 = Dropout(dropout)(u8)
    c8_1 = conv2d_block(u8_1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u9_1 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8_1)
    u9_1 = concatenate(axis=-1)([u9_1, c1])
    # u9 = Dropout(dropout)(u9)
    c9_1 = conv2d_block(u9_1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)


    u6_2 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6_2 = concatenate(axis=-1)([u6_2, c4])
    # u6 = Dropout(dropout)(u6)
    c6_2 = conv2d_block(u6_2, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    u7_2 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6_2)
    u7_2 = concatenate(axis=-1)([u7_2, c3])
    # u7 = Dropout(dropout)(u7)
    c7_2 = conv2d_block(u7_2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u8_2 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7_2)
    u8_2 = concatenate(axis=-1)([u8_2, c2])
    # u8 = Dropout(dropout)(u8)
    c8_2 = conv2d_block(u8_2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u9_2 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8_2)
    u9_2 = concatenate(axis=-1)([u9_2, c1])
    # u9 = Dropout(dropout)(u9)
    c9_2 = conv2d_block(u9_2, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    u6_3 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6_3 = concatenate(axis=-1)([u6_3, c4])
    # u6 = Dropout(dropout)(u6)
    c6_3 = conv2d_block(u6_3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    u7_3 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6_3)
    u7_3 = concatenate(axis=-1)([u7_3, c3])
    # u7 = Dropout(dropout)(u7)
    c7_3 = conv2d_block(u7_3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u8_3 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7_3)
    u8_3 = concatenate(axis=-1)([u8_3, c2])
    # u8 = Dropout(dropout)(u8)
    c8_3 = conv2d_block(u8_3, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u9_3 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8_3)
    u9_3 = concatenate(axis=-1)([u9_3, c1])
    # u9 = Dropout(dropout)(u9)
    c9_3 = conv2d_block(u9_3, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    o1 = Conv2D(number_class, (1, 1), activation='sigmoid')(c9_1)
    o2 = Conv2D(number_class, (1, 1), activation='sigmoid')(c9_2)
    o3 = Conv2D(number_class, (1, 1), activation='sigmoid')(c9_3)
    #1 is number of class
    model = Model(inputs=[input_img], outputs=[o1,o2,o3])
    return model

def get_unet_3classes_2(input_img, n_filters = 32, dropout = 0.1, batchnorm = True):
    
    """Function to define the UNET Model"""
    # Contracting Path    
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    o1 = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    o2 = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    o3 = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    #1 is number of class
    model = Model(inputs=[input_img], outputs=[o1,o2,o3])
    return model


def get_unet(input_img,number_class, n_filters = 32, dropout = 0.1, batchnorm = True):
    
    """Function to define the UNET Model"""
    # Contracting Path    
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c9)
    #1 is number of class
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def get_unet16_5(n_class, input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(n_class, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def get_unet2(n_class, input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(n_class, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_unet_32_5(input_img, n_filters = 32, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_unet_8_5(input_img, n_filters = 8, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



# =============================================================================
# Next Gen
# =============================================================================
def get_unet16_4(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#    p4 = MaxPooling2D((2, 2))(c4)
#    p4 = Dropout(dropout)(p4)
#    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u6 = concatenate(axis=-1)([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#    u9 = concatenate(axis=-1)([u9, c1])
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def get_unet32_4(input_img, n_filters = 32, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#    p4 = MaxPooling2D((2, 2))(c4)
#    p4 = Dropout(dropout)(p4)
#    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u6 = concatenate(axis=-1)([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#    u9 = concatenate(axis=-1)([u9, c1])
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def get_unet8_4(input_img, n_filters = 8, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#    p4 = MaxPooling2D((2, 2))(c4)
#    p4 = Dropout(dropout)(p4)
#    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u6 = concatenate(axis=-1)([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#    u9 = concatenate(axis=-1)([u9, c1])
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# =============================================================================
#  Nex gen
# =============================================================================
def get_unet8_6(input_img, n_filters = 8, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)    
    
    c00 = conv2d_block(p5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c00)
    u6 = concatenate(axis=-1)([u6, c5])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c4])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c3])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c2])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    u10 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u10 = concatenate(axis=-1)([u10, c1])
    u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# =============================================================================
#  Multi Classes
# =============================================================================
def get_munet16_5(input_img,number_class, n_filters = 16, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def get_munet_32_5(input_img,number_class, n_filters = 32, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def get_munet_8_5(input_img,number_class, n_filters = 8, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# =============================================================================
# Next Gen
# =============================================================================
def get_munet16_4(input_img,number_class, n_filters = 16, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#    p4 = MaxPooling2D((2, 2))(c4)
#    p4 = Dropout(dropout)(p4)
#    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u6 = concatenate(axis=-1)([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#    u9 = concatenate(axis=-1)([u9, c1])
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def get_munet32_4(input_img,number_class, n_filters = 32, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#    p4 = MaxPooling2D((2, 2))(c4)
#    p4 = Dropout(dropout)(p4)
#    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u6 = concatenate(axis=-1)([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#    u9 = concatenate(axis=-1)([u9, c1])
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def get_munet8_4(input_img,number_class, n_filters = 8, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#    p4 = MaxPooling2D((2, 2))(c4)
#    p4 = Dropout(dropout)(p4)
#    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u6 = concatenate(axis=-1)([u6, c3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c2])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c1])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#    u9 = concatenate(axis=-1)([u9, c1])
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# =============================================================================
#  Nex gen
# =============================================================================
def get_munet8_6(input_img,number_class, n_filters = 8, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)    
    
    c00 = conv2d_block(p5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    u6 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c00)
    u6 = concatenate(axis=-1)([u6, c5])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c4])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c3])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate(axis=-1)([u9, c2])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    u10 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u10 = concatenate(axis=-1)([u10, c1])
    u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    
    outputs = Conv2D(number_class, (1, 1), activation='sigmoid')(c10)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# =============================================================================
# NEW
# =============================================================================
# def MSTGANet_model 
import tensorflow as tf
class CustomLayer(tf.keras.layers.Layer):
  def __init__(self, n_dims, height = 16, width = 16):
    super(CustomLayer, self).__init__()
    self.height = height
    self.width  = width
    self.n_dims = n_dims    
    
  def get_config(self):
    
    config = super(CustomLayer, self).get_config()
    config['n_dims'] = self.n_dims
    config['height'] = self.height
    config['width']  = self.width
    return config    
  def build(self, input_shape):
    self.w1 = self.add_weight(shape=(self.height, 1, self.n_dims//8),
                             initializer='ones',
                             trainable=True, 
                             name = "w1")
    
    self.w2 = self.add_weight(shape=(1, self.width , self.n_dims//8),
                             initializer='ones',
                             trainable=True, 
                             name = "w2")    

  def call(self, inputs):      
      content_position  = self.w1 + self.w2            
      content_position = inputs * content_position      
      return content_position
  
# In[]:
class CustomLayer_2(tf.keras.layers.Layer):

  def __init__(self, size):
    super(CustomLayer_2, self).__init__()
    self.size = size
    
  def build(self, input_shape):
    self.w1 = self.add_weight(shape=(self.size[0], 1, 1),
                             initializer='ones',
                             trainable=True,
                             name = "w3")
    
    self.w2 = self.add_weight(shape=(1, self.size[1], 1),
                             initializer='ones',
                             trainable=True,
                             name = "w4")    

  def get_config(self):
    
      config = super(CustomLayer_2, self).get_config()
      config['size'] = self.size
      # config.update({
      #       'size': self.size,
      #   })
      return config

  def call(self, inputs):      
      content_position  = self.w1 + self.w2   
      content_position = inputs * content_position   
      return content_position
  
# In[]:
class Gamma(tf.keras.layers.Layer):

  def __init__(self):
    super(Gamma, self).__init__()
        
  def build(self, input_shape):
    self.w1 = self.add_weight(shape=(1,1,1),
                             initializer='ones',
                             trainable=True)

  def call(self, outs, x):                   
      out  = self.w1 * outs + x
      return out 
 
    
def spp_q(input_tensor, filters, height, width, ks=3):      
    x = tf.keras.layers.experimental.preprocessing.Resizing(height, width)(input_tensor)    
    x = tf.keras.layers.Conv2D(filters, (ks,ks), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    return x    

def encoder_pos(x_input,x_list, n_dims, filters):     
    m_batchsize, width, height, C = x_input.shape      
    x1 = spp_q(x_list[0], n_dims, height = int(x_list[0].shape[1] / 16), width = int(x_list[0].shape[2] / 16))
    x2 = spp_q(x_list[1], n_dims, height = int(x_list[1].shape[1] / 8) , width = int(x_list[1].shape[2] / 8))
    x3 = spp_q(x_list[1], n_dims, height = int(x_list[2].shape[1] / 4) , width = int(x_list[2].shape[2] / 4))    
    x4 = spp_q(x_list[3], n_dims, height = int(x_list[3].shape[1] / 2) , width = int(x_list[3].shape[2] / 2))
    x = x1 + x2 + x3 + x4    
    proj_query  = tf.keras.layers.Conv2D(n_dims // 8, (1,1), padding = "same")(x)
    proj_key    = tf.keras.layers.Conv2D(n_dims // 8, (1,1), padding = "same")(x_input)    
    energy_content  = tf.keras.layers.Multiply()([proj_query, proj_key])        
    content_position = CustomLayer(n_dims = n_dims)(energy_content)
    energy = tf.keras.layers.Add()([energy_content , content_position])
    attention   = tf.keras.layers.Activation("softmax")(energy)    
    attention   = tf.keras.layers.Conv2D(n_dims, (1,1), padding = "same")(attention)
    proj_value  = tf.keras.layers.Conv2D(n_dims, (1,1), padding = "same")(x_input)
    out = tf.keras.layers.Multiply()([proj_value, attention])      
    out = Gamma()(out, x_input)    
    return out, attention

def decoder_pos(x, x_encoder, n_dims):    
    proj_query  = tf.keras.layers.Conv2D(n_dims // 8, (1,1), padding = "same")(x)
    proj_key    = tf.keras.layers.Conv2D(n_dims // 8, (1,1), padding = "same")(x_encoder)
    energy_content  = tf.keras.layers.Multiply()([proj_query, proj_key])        
    content_position = CustomLayer(n_dims = n_dims)(energy_content)
    energy = tf.keras.layers.Add()([energy_content , content_position])
    attention   = tf.keras.layers.Activation("softmax")(energy)    
    attention   = tf.keras.layers.Conv2D(n_dims, (1,1), padding = "same")(attention)
    proj_value  = tf.keras.layers.Conv2D(n_dims, (1,1), padding = "same")(x_encoder)
    out = tf.keras.layers.Multiply()([proj_value, attention])      
    out = Gamma()(out, x)    
    return out

def MsTNL(x_input, x_list,n_dims,filters):    
    out, attention = encoder_pos(x_input, x_list, n_dims, filters)
    out = decoder_pos(out, attention, n_dims)
    return out

def upconv(input_tensor, filters):
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(input_tensor)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)        
    return x

def conv_block(input_tensor, filters):    
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same")(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)        
    return x

def MsGCS(g, x, F_int, size):
    input_layer = tf.keras.layers.concatenate([g, x])
    x = tf.keras.layers.Conv2D(F_int, (3,3), padding = "same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(F_int, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_Multi_Scale      = tf.keras.layers.Activation('relu')(x)     
    x_att_multi_scale  = CustomLayer_2(size = size)(x_Multi_Scale)
    x_att_multi_scale = tf.keras.layers.BatchNormalization()(x_att_multi_scale)
    x_att_multi_scale = tf.keras.layers.Activation('softmax')(x_att_multi_scale)       
    out = x_Multi_Scale * x_att_multi_scale
    return out





            # (input_img                ,number_class, n_filters = 32, dropout = 0.1, batchnorm = True)

def get_model_MSTGANet_model1 (input_shape1 = (256,256,1)):
    feature_scale = 1
    filters = [64, 128, 256, 512, 1024]
    filters = [int(x / feature_scale) for x in filters]
    print('input_shape1 =',input_shape1)
    # input_layer = tf.keras.layers.Input(input_shape1)
    input_layer = input_shape1
    x1 = conv_block(input_layer, filters[0])
    x2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x1) 
    x2 = conv_block(x2, filters[1])
    x3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x2)         
    x3 = conv_block(x3, filters[2])
    x4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x3)           
    x5 = conv_block(x4, filters[3])
    x5 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x4)         
    x5 = conv_block(x5, filters[4])
    x5 = MsTNL(x5, [x1, x2, x3, x4], n_dims = 512, filters = filters)
    #Decoder1         
    d5 = upconv(x5, filters[3])    
    x4 = MsGCS(d5, x4, filters[2], size=(32, 32))        
    d5 = tf.keras.layers.concatenate([x4, d5])    
    d5 = conv_block(d5, filters[3])
    #Decoder2
    d4 = upconv(d5, filters[2])    
    x3 = MsGCS(d4, x3, filters[1], size=(64,64))        
    d4 = tf.keras.layers.concatenate([x3, d4])    
    d4 = conv_block(d4, filters[2])
    #Decoder3
    d3 = upconv(d4, filters[1])    
    x2 = MsGCS(d3, x2, filters[0], size=(128,128))        
    d3 = tf.keras.layers.concatenate([x2, d3])    
    d3 = conv_block(d3, filters[1])
    #Decoder4
    d2 = upconv(d3, filters[0])    
    x1 = MsGCS(d2, x1, filters[0] // 2, size=(256,256))        
    d2 = tf.keras.layers.concatenate([x1, d2])    
    d2 = conv_block(d2, filters[0])    
            
    c9 = tf.keras.layers.Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'softmax')(d2)
                                      
    # model_ = tf.keras.models.Model(inputs = [input_layer], outputs = [output])
    
    
    
    o1 = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    o2 = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    o3 = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    #1 is number of class
    model_ = Model(inputs=[input_layer], outputs=[o1,o2,o3])
    
    
    return model_





def forward_att(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x = concatenate(axis=-1)([x, x1])

    return x

def get_model_MSTGANet_model(input_img, n_filters = 32, dropout = 0.1, batchnorm = True):
    
    """Function to define the UNET Model"""
    # Contracting Path    
    c0 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    # c1 = Conv2D(1, (1, 1), activation='sigmoid')(p1)
    
    c1 = Conv2D(filters = 1, kernel_size = (1, 1),\
              kernel_initializer = 'he_normal', padding = 'same')(c0)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)    
        

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    p1 = Conv2D(1, (1, 1), activation='sigmoid')(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = forward_att(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = forward_att(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate(axis=-1)([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = forward_att(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate(axis=-1)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = forward_att(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate(axis=-1)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = forward_att(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    
    c1_9 = concatenate(axis=-1)([c1, u9])
    c1_t = forward_att(c1_9, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    u9 = concatenate(axis=-1)([u9, c1_t])
    c1_att = conv2d_block(c1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    
    c2_8 = concatenate(axis=-1)([c2, c8])
    c2_t = forward_att(c2_8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    c2_1 = conv2d_block(c2_t, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2_att = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same', name='c1_att')(c2_1)
    c2_1_2 = conv2d_block(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2_att_2 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c2_1_2)
    
    
    c3_7 = concatenate(axis=-1)([c3, c7])
    c3_t = forward_att(c3_7, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    
    uu9 = concatenate(axis=-1)([c1_att, c2_att,  c2_att_2 ])


    u9 = concatenate(axis=-1)([u9, uu9, c1_t ])
    
    
    # u9 = concatenate(axis=-1)([u9, uu9 ])
    # u9 = concatenate(axis=-1)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block2(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    out_title =['Right_Ventricle' , 'Myocard' ,'Left_Ventricle']

    o1 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[0])(c9)
    o2 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[1])(c9)
    o3 = Conv2D(1, (1, 1), activation='sigmoid',name= out_title[2])(c9)
    #1 is number of class
    model = Model(inputs=[input_img], outputs=[o1,o2,o3])
    return model
