"""   Created on Wed May 17 01:33:04 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""


# def function_predictor_dual(img1,mask1):



def function_predictor1(img1,mask1):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import os 
    from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
    from Models_unet import get_model_MSTGANet_model
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # =============================================================================
    #  def
    # =============================================================================
    def Grad_CAM1 (model,X,index=0,tavan=1, tavan2=1 , smoothed=True,alpha=0.4,alpha2=0.6):
        # plt.subplot(2,3,1);plt.imshow(X[index],cmap='gray' );plt.title('Original image')
        X3 = np.zeros ( [np.shape(X[index])[0] ,  np.shape(X[index])[1],3])
        X3[:,:,0] =Y1[index,:,:,0]
        X3[:,:,1] =Y2[index,:,:,0]
        X3[:,:,2] =Y3[index,:,:,0]
        # plt.subplot(2,3,2);plt.imshow(X3,cmap='gray' );plt.title('Original Masks')
        index_train=range(index+1)
        index_train=[index,index]
        X=X[index_train];
        e=0.00001
        cnt=0
        # X=X[index]
        for layer in model.layers:
            cnt=cnt+1
        last_conv_layer_name1 =model.layers[-1].name
        last_conv_layer_name2 =model.layers[-2].name
        last_conv_layer_name3 =model.layers[-3].name
        
        p01,p02,p03 = model.predict(X , verbose=1)
        
        # index=2
        XX1,YY1  =  masker (X,0,p01,p02,p03)
        # plt.subplot(2,3,3);plt.imshow( YY1 ,cmap='gray' );plt.title('Predicted Masks \n'+'(Without Activation Function)')
        
        model.layers[-1].activation = None
        model.layers[-3].activation = None
        model.layers[-2].activation = None
        
        grad_model1 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name1).output])
        grad_model2 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name2).output])
        grad_model3 = tf.keras.models.Model(    [model.inputs], [model.get_layer(last_conv_layer_name3).output])
        
        
        p1   = grad_model1.predict(X , verbose=1)
        p2   = grad_model2.predict(X , verbose=1)
        p3   = grad_model3.predict(X , verbose=1)
        
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
        # plt.figure();plt.imshow( p ,cmap='jet' )
        p=np.uint8( (255)*(p-np.min(p)-e)/ (np.max(p)-np.min(p)-e) );
        # plt.figure();
        # plt.subplot(2,3,4);    plt.imshow(  p ,cmap='jet' );plt.title('Grad-CAM')
        X1 = X[0]
        X3 = np.zeros ( [np.shape(X1)[0] ,  np.shape(X1)[1],3])
        X3[:,:,0]=X1[:,:,0];X3[:,:,1]=X1[:,:,0];X3[:,:,2]=X1[:,:,0]
        # plt.figure();plt.imshow( X3 ,cmap='gray' )
        pp = cv2.cvtColor(p,cv2.COLOR_GRAY2RGB)
        heatmap = cv2.cvtColor(pp,cv2.COLOR_RGB2GRAY)
        
        
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
        # plt.subplot(2,3,5);plt.imshow(superimposed_img1 ) # plt.figure();plt.imshow(superimposed_img2 ) 
        # plt.title(' Superimposed Visualization')
        jet_heatmap=jet_heatmap*jetgray_heatmap
        # plt.figure();plt.imshow(jet_heatmap)# Create an image with RGB colorized heatmap
        jet_heatmap2=1*(jet_heatmap-np.min(jet_heatmap)-e)/ (np.max(jet_heatmap)-np.min(jet_heatmap)-e);
        superimposed_img = jet_heatmap2 * alpha + img* alpha2 # superimposed_img2 = (jet_heatmap2 * 0.6)* (img)
        # plt.subplot(2,3,6);plt.imshow(superimposed_img ) # plt.figure();plt.imshow(superimposed_img2 ) 
        # plt.title(' Superimposed Visualization2')
        
        return YY1 , p,superimposed_img1, superimposed_img
        
    
        
    
    from saed import masker
    GRAD_CAM_FOLDER=2
    plt.close("all")
    
    GRAD_CAM_FOLDER='GRAD_CAM_FOLDER'
    try:os.mkdir(GRAD_CAM_FOLDER)
    except:s=1
    
    
    # os.startfile( GRAD_CAM_FOLDER)
    # path0=os.getcwd()
    # data1 = np.load('DATA.npz', allow_pickle = True)
    im_width=128;im_height=128
    # file_name   = 'input3_'+str(im_width)+'_'+str(im_height)+'.npz'
    # data1 = np.load(file_name ,allow_pickle=True )
    # X  = data1['X'] ;Y1 = data1['Y1'];Y2 = data1['Y2'];Y3 = data1['Y3']
    input_img = Input((im_width, im_height, 1), name='img')
    
    # zzzz
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img    
    from skimage.transform import resize
    
    import cv2
    
    x_img = img_to_array(img1)
    mask_img = img_to_array(mask1)
     
    x_img = resize(x_img, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
    mask_img = resize(mask_img, (im_width, im_height,1), mode = 'constant', preserve_range = True)
      
    # X   = np.zeros((len(image), im_height, im_width, 1), dtype=np.float32)
    X   = np.zeros((2, im_height, im_width, 1), dtype=np.float32)
    X[1] = x_img/255.0;
    X[0] = x_img/255.0;
    Y1   = np.zeros((2, im_height, im_width,1), dtype=np.float32)
    Y2   = np.zeros((2, im_height, im_width,1), dtype=np.float32)
    Y3   = np.zeros((2, im_height, im_width,1), dtype=np.float32)
    
    Y1[1,:,:,0] = mask_img[:,:,0]/255.0;
    Y1[0,:,:,0] = mask_img[:,:,0]/255.0;
    Y2[1,:,:,0] = mask_img[:,:,0]/255.0;
    Y2[0,:,:,0] = mask_img[:,:,0]/255.0;
    Y3[1,:,:,0] = mask_img[:,:,0]/255.0;
    Y3[0,:,:,0] = mask_img[:,:,0]/255.0;
    
    
    index=0  
    
    n_filters1=10
    weight_model_path='my models vit2\\WB_MSTGANET_model_best3_128_12810_kfold = 9.h5'
    model = get_model_MSTGANet_model (input_img, n_filters = n_filters1, dropout = 0.05, batchnorm = True)
    model.load_weights (weight_model_path)
    weight_model_path1=weight_model_path
    
    
    YY1 , p,superimposed_img1 , superimposed_img =Grad_CAM1(model,X,index)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.pause(1);
    # plt.savefig( GRAD_CAM_FOLDER +'\\'+weight_model_path1+'_index = '+str(index)+'.png')
    # plt.close("all")
    
    return YY1 , p, superimposed_img1, superimposed_img
   
 
# example 1
# ssss
# import cv2
# import matplotlib.pyplot as plt
# img1  =cv2.imread ('dataset\images1\P0_0_2.png')[:,:,0]
# mask1 =cv2.imread ('dataset\masks1\P0_2.png')[:,:,0] 
# YY1 , p, superimposed_img1, superimposed_img =function_predictor1(img1,mask1)
# plt.figure()
# plt.subplot(2,3,1);plt.imshow( img1 ,cmap='gray' );plt.title('img1)')
# plt.subplot(2,3,2);plt.imshow( mask1 ,cmap='gray' );plt.title('mask1')
# plt.subplot(2,3,3);plt.imshow( YY1 ,cmap='gray' );plt.title('Y')
# plt.subplot(2,3,4);plt.imshow( p ,cmap='jet' );plt.title('p')
# plt.subplot(2,3,5);plt.imshow( superimposed_img1 ,cmap='jet' );plt.title('superimposed_img')
# plt.subplot(2,3,6);plt.imshow( superimposed_img ,cmap='gray' );plt.title('superimposed_img')


