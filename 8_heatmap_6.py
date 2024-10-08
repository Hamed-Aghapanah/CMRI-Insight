import numpy as np
import tensorflow as tf
from tensorflow import keras
import os 

# Display
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
#  def
# =============================================================================
def masker (X,index,Y1,Y2,Y3) :
    X1=X[index]
    X3 = np.zeros ( [np.shape(X1)[0] ,  np.shape(X1)[1],3])
    X3[:,:,0]=X1 [:,:,0];X3[:,:,1]=X1 [:,:,0];X3[:,:,2]=X1 [:,:,0]
    YY=0*X3;
    
    YY[:,:,0]=Y1 [index,:,:,0];
    YY[:,:,1]=Y2 [index,:,:,0];
    YY[:,:,2]=Y3 [index,:,:,0]
    XX=X3
    for i in range(np.shape(XX)[0]):
        for j in range(np.shape(XX)[1]):
            for k in range(np.shape(XX)[2]):
                if YY[i,j,k]>0:     XX[i,j,:]=0;XX[i,j,k]=1
    return XX,YY
 
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# =============================================================================
# 
# =============================================================================
# model_builder = keras.applications.xception.Xception
img_size = (128, 128)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions


plt.close("all")
path0=os.getcwd()
data1 = np.load('DATA.npz', allow_pickle = True)

im_width=128
im_height=128
file_name   = 'input3_'+str(im_width)+'_'+str(im_height)+'.npz'
data1 = np.load(file_name ,allow_pickle=True )
X  = data1['X'] 
Y1 = data1['Y1']
Y2 = data1['Y2']
Y3 = data1['Y3']


# mask = data1['mask']
# image = data1['image']

plt.subplot(131);plt.imshow(X[1],cmap='gray' )
plt.subplot(132);plt.imshow(0.3*Y1[1] +0.6*Y2[1]+ Y3[1],cmap='gray' )
# plt.subplot(223);plt.imshow(Y2[1],cmap='gray' )
# plt.subplot(224);plt.imshow(Y3[1],cmap='gray' )
# display(Image(img_path))

# Prepare image
index_train=range(5)
X=X[index_train];
# X_train0 = np.concatenate([X_train0]*3, -1)


# Make model
# model = model_builder(weights="imagenet")
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout

from Models_unet import get_model_MSTGANet_model
input_img = Input((128, 128, 1), name='img')
n_filters1=10
model = get_model_MSTGANet_model (input_img, n_filters = n_filters1, dropout = 0.05, batchnorm = True)
model.load_weights ('WB_MSTGANET_model_best3_128_12810_kfold = 9.h5')
# model.save('saed.h5')
# os.startfile('saed.h5')


cnt=0
for layer in model.layers:
    cnt=cnt+1
    
    # if cnt<=50 and 'batch_' in layer.name: 
    #     s= layer.name
    # print(cnt,layer.name)
    
last_conv_layer_name1 =layer.name
last_conv_layer_name1 =model.layers[-1].name
last_conv_layer_name2 =model.layers[-2].name
last_conv_layer_name3 =model.layers[-3].name
 # "block14_sepconv2_act"


# model=
# Remove last layer's softmax

# Print what the top predicted class is

p1,p2,p3 = model.predict(X , verbose=1)
index=2
XX1,YY1  =  masker (X,index,p1,p2,p3)
plt.figure();plt.imshow( YY1 ,cmap='gray' )
# Generate class activation heatmap
# for j in range(100):
#     model.layers[-1*j-1].activation = None
model.layers[-1].activation = None
model.layers[-3].activation = None
model.layers[-2].activation = None

p1,p2,p3 = model.predict(X , verbose=1)
index=2
XX1,YY1  =  masker (X,index,p1,p2,p3)
plt.figure();plt.imshow( YY1 ,cmap='gray' )


# grad_model1 = tf.keras.models.Model(
#     [model.inputs], [model.get_layer(last_conv_layer_name1).output])

# grad_model2 = tf.keras.models.Model(
#     [model.inputs], [model.get_layer(last_conv_layer_name2).output])

# grad_model3 = tf.keras.models.Model(
#     [model.inputs], [model.get_layer(last_conv_layer_name3).output])

# grad_model.save('saed.h5')
# os.startfile('saed.h5')


# p1   = grad_model1.predict(X , verbose=1)
p1,p2,p3   = model.predict(X , verbose=1)
# p1,p2,p3 = model.predict(X_train0)
preds=p1[1]
# print("Predicted:", decode_predictions(preds, top=1)[0])
# plt.subplot(133);plt.imshow( preds ,cmap='gray' )
index=2
# XX1,YY1  =  masker (X,index,p1,p2,p3)
p11=  (p1[0][index] )
p22=  (p1[1][index] )
p33=  (p1[2][index] )
plt.figure();plt.imshow( p11 ,cmap='gray' )
plt.figure();plt.imshow( p22 ,cmap='gray' )
plt.figure();plt.imshow( p33 ,cmap='gray' )


"""
img_array=X[index]
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# First, we create a model that maps the input image to the activations
# of the last conv layer as well as the output predictions
grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
)

grad_model.save('saed.h5')
os.startfile('saed.h5')


ss
# Then, we compute the gradient of the top predicted class for our input image
# with respect to the activations of the last conv layer

with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    if pred_index is None:
        pred_index = tf.argmax(preds[0])
    class_channel = preds[:, pred_index]

# This is the gradient of the output neuron (top predicted or chosen)
# with regard to the output feature map of the last conv layer
grads = tape.gradient(class_channel, last_conv_layer_output)

# This is a vector where each entry is the mean intensity of the gradient
# over a specific feature map channel
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the top predicted class
# then sum all the channels to obtain the heatmap class activation
last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# For visualization purpose, we will also normalize the heatmap between 0 & 1
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
# return heatmap.numpy()



# Display heatmap
plt.figure();
plt.imshow(heatmap)
plt.show()



"""