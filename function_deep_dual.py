"""   Created on Fri May 26 23:05:35 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""

def function_deep_dual (I3 , mask1=3):
    plot_enable=0
    import numpy as np
    if np.size(mask1)==1:
        mask1 =0*I3
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import os 
    from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
    from Models_unet import get_model_MSTGANet_model
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from model_CardSegNet2 import BBVDAMLF_model 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img    
    from skimage.transform import resize
    # =============================================================================
    #  def
    # =============================================================================
    def roundd(a=123.456,b=1):  # a=0.123456789;   # b=3
        aa = a*100*10**b;
        aa= int (aa);
        aa=aa/(10**b)
        return aa
    
    
    DICE_ALL=[]
    from saed import masker
    from saed import Grad_CAM1
    from saed import Grad_CAM2
    GRAD_CAM_FOLDER=2
    if plot_enable==1:
        plt.close("all")
    
    GRAD_CAM_FOLDER='GRAD_CAM_FOLDER'
    try:os.mkdir(GRAD_CAM_FOLDER)
    except:s=1
    
    just_one=1
    indexes=range(200)
    # os.startfile( GRAD_CAM_FOLDER)
    # path0=os.getcwd()
    # data1 = np.load('DATA.npz', allow_pickle = True)
    im_width=128;im_height=128
    # file_name   = 'input3_'+str(im_width)+'_'+str(im_height)+'.npz'
    # data1 = np.load(file_name ,allow_pickle=True )
    # X  = data1['X'] ;Y1 = data1['Y1'];Y2 = data1['Y2'];Y3 = data1['Y3']
    file_name   = 'input3_'+str(im_width)+'_'+str(im_height)+'.npz'
    # ss
    # if  file_name in files:
        # if show_print:
            # print('file is existance')
    # data1 = np.load(file_name ,allow_pickle=True )
    # X  = data1['X']
    
    # XXX = np.load('X3.npz', allow_pickle = True)
    # X3 = XXX['X3']
    
    # Y1 = data1['Y1']
    # Y2 = data1['Y2']
    # Y3 = data1['Y3']
    
    
    x_img = img_to_array(I3)
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
    
    # zzzz
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img    
    from skimage.transform import resize
    
    import cv2
    
    
    
    
    input_img = Input((im_height, im_width, 1), name='img')
    input_img3 = Input((im_height, im_width, 3), name='img')
    n_filters1=10
     
    
    # weight_model_path='my models vit2\\WB_MSTGANET_model_best3_128_12810_kfold = 9.h5'
    # model = get_model_MSTGANet_model (input_img, n_filters = n_filters1, dropout = 0.05, batchnorm = True)
    # model.load_weights (weight_model_path)
    # weight_model_path1=weight_model_path
    
    
    # weight_model_path=os.getcwd()+'//loading_segmentation_model//weight217.h5'
    # model1=BBVDAMLF_model(input_img3,bb_name='resnet',  dual_attention_enable_model='vsc',
    #                       n_filters = 10, dropout = 0.1, batchnorm = True)
    # model1.load_weights (weight_model_path)    
    
    finding=0
    dual_attention_enable_modelss=['vsc ','vs  ','v c ',' sc ', 'v   ','  c ',' s  ','none',]
    bb_enable_modelss=['resnet','none']
    weight_model_path=os.getcwd()+'//loading_segmentation_model//wfinal_model resnet DAvsc   1time11_53_44.h5'
    weight_model_path=os.getcwd()+'//loading_segmentation_model//weight217.h5'
    weight_model_path=os.getcwd()+'//loading_segmentation_model//weight218.h5'
    # weight_model_path=os.getcwd()+'//loading_segmentation_model//wModel_temp.h5'
    # weight_model_path=os.getcwd()+'//loading_segmentation_model//Weight_best_128_12850.h5'
    
    
    
    weight_model_path=os.getcwd()+'//loading_segmentation_model//weight91.h5'
    weight_model_path=os.getcwd()+'//loading_segmentation_model//weight2.h5'
    weight_model_path=os.getcwd()+'//loading_segmentation_model//wModel_temp.h5'
    weight_model_path=os.getcwd()+'//loading_segmentation_model//WB_MSTGANET_model_best3_128_12810_kfold = 9.h5'
    
    
    
    weight_model_path=os.getcwd()+'//loading_segmentation_model//Wmodel_CENET bb_nonedual_none.h5'
    model_path=os.getcwd()+'\\loading_segmentation_model\\model_CENET bb_nonedual_none.h5'
    
    # os.startfile('Model_mode_dual_dual_vsc_.h5')
    # model1=load_model('Model_mode_dual_dual_vsc_.h5')
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K
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
    # model_path='DA_MSTGANET2_model_best3_128_12810_kfold = 9.h5'
    # os.startfile(model_path)
    from keras.models import load_model
    # from tensorflow.keras.models import load_model
    
    # model1=load_model(model_path)
    
    weight_model_path= 'WDA_MSTGANET2_model_best3_128_12810_kfold = 9.h5'
    path_save='res_GUI\\'
    weight_model_path= path_save +'wmodel_MSTGANet_last_' +str(1)+'.h5'

    # os.startfile(weight_model_path)
    cnt_dice=0
    if finding==0:
        
        # model_path=os.getcwd()+'//loading_segmentation_model//model_CENET bb_nonedual_none.h5'
        from Models_unet import get_model_MSTGANet_model
        # model1 = get_model_MSTGANet_model (input_img, n_filters = 8, dropout = 0.05, batchnorm = True)
        # from Models_unet import get_model_MSTGANet_model
        n_filters1=10
        weight_model_path='my models vit2\\WB_MSTGANET_model_best3_128_12810_kfold = 9.h5'
        model1 = get_model_MSTGANet_model (input_img, n_filters = n_filters1, dropout = 0.05, batchnorm = True)
        model1.load_weights (weight_model_path)
        # model1.load_weights (weight_model_path)
        # model1 = load_model(model_path)
        # model1.load_weights (weight_model_path)
        finding=1
        model2=model1
        model =model1
        # index=1
        # YY1 , p,superimposed_img1 , superimposed_img =Grad_CAM2(model2,X3,Y1,Y2,Y3,index)
        # sss
    # def Grad_CAM1 (model,X,Y1,Y2,Y3,index=0,tavan=1, tavan2=1 , smoothed=True,alpha=0.4,alpha2=0.6):
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
        # for i in indexes:
        index=0
        # X  = data1['X']
        X   = np.zeros((2, im_height, im_width, 1), dtype=np.float32)
        X[1] = x_img/255.0;
        X[0] = x_img/255.0;
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
    
        if plot_enable==1:
            plt.figure(1000)
            plt.subplot(2,3,1);plt.imshow(X[0],cmap='gray' );plt.title('Original image')
            plt.subplot(2,3,2);plt.imshow(Y,cmap='gray' );plt.title('Original Masks')
    
    
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
        p01[0,:,:,0]=ther(p01[0,:,:,0], np.mean(p01[0,:,:,0]))
        p02[0,:,:,0]=ther(p02[0,:,:,0], np.mean(p02[0,:,:,0]))
        p03[0,:,:,0]=ther(p03[0,:,:,0], np.mean(p03[0,:,:,0]))
        
        Y[:,:,0]=ther(Y[:,:,0], np.mean(Y[:,:,0]))
        Y[:,:,1]=ther(Y[:,:,1], np.mean(Y[:,:,1]))
        Y[:,:,2]=ther(Y[:,:,2], np.mean(Y[:,:,2]))
        
        
        def dot_multi(a=123.456,b=1):
            import copy
            c=copy.deepcopy(a*0)
            for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    c[i,j]=a[i,j]*b[i,j]
            return c
        def sumer(a=123.456 ):
            c=np.sum(a)
            c=np.sum(c)
            c=np.sum(c)
            c=int(c)
            return c
            
        def dot_add(a=123.456,b=1):
            import copy
            c=copy.deepcopy(a*0)
            for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    c[i,j]=a[i,j]+b[i,j]               
            return c
            
        XX1,YY1  =  masker (X,0,p01,p02,p03)
        
        d_r=1
        d_l=1
        d_m=1
        if sumer ( Y[:,:,0])>0 :
            d_r =int ( 100*2* sumer(dot_multi( Y[:,:,0],p01[0,:,:,0])) /sumer(( Y[:,:,0]+p01[0,:,:,0])))/100
        if sumer ( Y[:,:,1])>0 :
            d_l =int ( 100*2* sumer(dot_multi( Y[:,:,1],p02[0,:,:,0])) /sumer(( Y[:,:,1]+p02[0,:,:,0])))/100
        if sumer ( Y[:,:,2])>0 :
            d_m =int ( 100*2* sumer(dot_multi( Y[:,:,2],p03[0,:,:,0])) /sumer(( Y[:,:,2]+p03[0,:,:,0])))/100
        
        d_r= np.fix(d_r*100)/100
        d_l= np.fix(d_l*100)/100
        d_m= np.fix(d_m*100)/100
        
        
        if plot_enable==1:
            CMAP='jet'
            CMAP1='gray'
            CMAP='gray'
            
            plt.figure(2000)
            plt.subplot(3,4,1);plt.imshow(Y[:,:,0],cmap=CMAP);plt.ylabel(sumer(Y[:,:,0]))
            plt.subplot(3,4,2);plt.imshow(p01[0,:,:,0],cmap=CMAP);plt.title('Dice RV '+str( d_r));plt.ylabel(sumer(p01[0,:,:,0]))
            plt.subplot(3,4,3);plt.imshow(dot_multi( Y[:,:,0],p01[0,:,:,0]),cmap=CMAP);plt.title('intersection '+str(sumer(dot_multi( Y[:,:,0],p01[0,:,:,0]))))
            plt.subplot(3,4,4);plt.imshow(( Y[:,:,0]+p01[0,:,:,0]),cmap=CMAP);plt.title('sumerize '+str(sumer(( Y[:,:,0]+p01[0,:,:,0]))))
            
            plt.subplot(3,4,5);plt.imshow(Y[:,:,1],cmap=CMAP);plt.ylabel(sumer(Y[:,:,1]))
            plt.subplot(3,4,6);plt.imshow(p02[0,:,:,0],cmap=CMAP);plt.title('Dice LV '+str( d_l));plt.ylabel(sumer(p02[0,:,:,0]))
            plt.subplot(3,4,7);plt.imshow(dot_multi( Y[:,:,1],p02[0,:,:,0]),cmap=CMAP);plt.title('intersection '+str(sumer(dot_multi( Y[:,:,1],p02[0,:,:,0]))))
            plt.subplot(3,4,8);plt.imshow(( Y[:,:,1]+p02[0,:,:,0]),cmap=CMAP);plt.title('sumerize '+str(sumer(( Y[:,:,1]+p02[0,:,:,0]))))
            
            plt.subplot(3,4,9);plt.imshow(Y[:,:,2],cmap=CMAP);plt.ylabel(sumer(Y[:,:,0]))
            plt.subplot(3,4,10);plt.imshow(p03[0,:,:,0],cmap=CMAP);plt.title('Dice my '+str( d_m));plt.ylabel(sumer(p03[0,:,:,0]))
            plt.subplot(3,4,11);plt.imshow(dot_multi( Y[:,:,2],p03[0,:,:,0]),cmap=CMAP);plt.title('intersection '+str(sumer(dot_multi( Y[:,:,2],p03[0,:,:,0]))))
            plt.subplot(3,4,12);plt.imshow(( Y[:,:,2]+p03[0,:,:,0]),cmap=CMAP);plt.title('sumerize '+str(sumer(( Y[:,:,2]+p03[0,:,:,0]))))
        # fig = plt.figure()
        # plt.plot(range(10))
        
        try:
            os.mkdir('res_dice')
        except:s=1
        if plot_enable==1:
            fig = plt.figure(2000)
            cnt_dice=cnt_dice+1
            fig.savefig('res_dice\\temp'+str(cnt_dice)+'.png' )
            plt.close(2000)
        # sss
        
        d_overal= int ( 100*(d_r+d_l+d_m))/300
        d_overal= np.fix(d_overal*100)/100
        
        ddd=str(d_r) +' ,'+str(d_l) +' , '+str(d_m) +' , '+str(d_overal) 
        if plot_enable==1:
            plt.subplot(2,3,3);plt.imshow( YY1 ,cmap='gray' );plt.title('Predicted Masks \n'+'dice ='+ddd)
        
        # xxxx
        last_conv_layer_name1 =model.layers[-1].name
        last_conv_layer_name2 =model.layers[-2].name
        last_conv_layer_name3 =model.layers[-3].name
        # for i in range(20):
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
        if plot_enable==1:
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
        
        jet = cm.get_cmap("gist_rainbow")
        jet = cm.get_cmap("rainbow")
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
        if plot_enable==1:
            plt.subplot(2,3,5);plt.imshow(superimposed_img1 ) # plt.figure();plt.imshow(superimposed_img2 ) 
            plt.title(' Superimposed Visualization')
        jet_heatmap=jet_heatmap*jetgray_heatmap
        # plt.figure();plt.imshow(jet_heatmap)# Create an image with RGB colorized heatmap
        jet_heatmap2=1*(jet_heatmap-np.min(jet_heatmap)-e)/ (np.max(jet_heatmap)-np.min(jet_heatmap)-e);
        superimposed_img = jet_heatmap2 * alpha + img* alpha2 # superimposed_img2 = (jet_heatmap2 * 0.6)* (img)
        if plot_enable==1:
            plt.subplot(2,3,6);plt.imshow(superimposed_img ) # plt.figure();plt.imshow(superimposed_img2 ) 
            plt.title(' Superimposed Visualization2')
        
        try:
            os.mkdir('res_dice')
        except:s=1
        if plot_enable==1:
            fig = plt.figure(1000)
            fig.savefig('res_dice\\Grad_'+str(cnt_dice)+'.png' )
            plt.close(1000)
        # if just_one==1:
        #     just_one=0
        #     os.startfile('res_dice')
            
        DICE_ALL.append(d_overal)
        if plot_enable==1:
            plt.figure(5000)
            plt.plot(DICE_ALL)          
        
    # YY1 , p, superimposed_img1, superimposed_img 
    return  YY1 , p, superimposed_img1, superimposed_img