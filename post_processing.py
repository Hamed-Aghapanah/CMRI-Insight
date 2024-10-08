"""   Created on Tue Jun 13 10:48:33 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""
def post_processing (image3,show=0):
    import numpy as np
    import cv2 
    import matplotlib.pyplot as plt
    import os 
    RV  = image3[:,:,0]
    Myo = image3[:,:,1]
    LV  = image3[:,:,2]
    BG   = 1-np.sum(image3,2)
    index_RV=np.where  (RV>0)
    index_Myo=np.where (Myo>0)
    index_LV=np.where  (LV>0)

    if np.sum(index_RV)>0  and np.sum(index_Myo)>0  and np.sum(index_LV)>0 :
        # condition 0 maximum partion
        if show>0:
            plt.figure(2)
            plt.subplot(241);plt.imshow( RV) ;plt.grid(False)
            plt.subplot(242);plt.imshow( Myo) ;plt.grid(False)
            plt.subplot(243);plt.imshow( LV) ;plt.grid(False)
            plt.subplot(244);plt.imshow( BG) ;plt.grid(False)
            plt.grid(False)
        # end show 


        import copy
        # condition 0    maximum segment

        image = copy.deepcopy(LV)
        import imutils
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        LV1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(Myo)
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        Myo1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(RV)
        RV=0*RV
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        RV1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(BG)
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        BG1=copy.deepcopy(image[:,:,0])




        # condition 1 LV is not connect to RV
        il,jl = np.where (LV1)
        iR,jR = np.where (RV1)
        im,jm = np.where (Myo1)

        for i in range(len (il)):
            for j in range(len (iR)):
                diss=np.sqrt( (il[i] -iR[j]  )**2 +( jl[i] -jR[j]  )**2 )
                if diss<2:
                    LV1 [ il[i] , jl[i]  ] =0
                    Myo1[ il[i] , jl[i]  ]=np.max(Myo1)
                
        # condition 2 between RV and LV   we have not B just Myo 

        if np.sum(Myo1)>0:
            for i in range(3,np.shape(Myo1)[0] -3):
                for j in range(3,np.shape(Myo1)[1] -3):
                    if np.mean(Myo1[i-2:i+2 , j-2:j+2]) > 0.8*np.max(Myo1):
                        Myo1[ i , j ]=np.max(Myo1)
        if np.sum(LV1)>0:
            for i in range(3,np.shape(LV1)[0] -3):
                for j in range(3,np.shape(LV1)[1] -3):
                    if np.mean(LV1[i-2:i+2 , j-2:j+2]) >0.8*np.max(LV1):
                        LV1[ i , j ]=np.max(LV1)
        if np.sum(RV1)>0:
            for i in range(3,np.shape(RV1)[0] -3):
                for j in range(3,np.shape(RV1)[1] -3):
                    if np.mean(RV1[i-2:i+2 , j-2:j+2]) >0.8*np.max(RV1):
                        RV1[ i , j ]=np.max(RV1)

        # condition 3 in myo we have BG (so subscrub LV from myo)
        il,jl = np.where (LV1)
        iR,jR = np.where (RV1)
        im,jm = np.where (Myo1)
        for i in range(len (il)):
            for j in range(len (Myo1)):
                Myo1[ il[i] , jl[i]  ]=0



        # condition 4 RV are connected to Myo (if existence)
        il,jl = np.where (LV1)
        iR,jR = np.where (RV1)
        im,jm = np.where (Myo1)
        connected=False
        if len(im) >1 and len(iR) >1 :
            dis=[];pos=[];
            dis_min=5000;
            for i in range(len (im)):
                for j in range(len (iR)):
                    diss=np.sqrt( (im[i] -iR[j]  )**2 +( jm[i] -jR[j]  )**2 )
                    dis_min=np.max([dis_min ,6])
                    if diss<dis_min  and diss>6 :
                        dis_min=diss
                        dis=[];pos=[];
                    if diss==dis_min  :
                        dis.append(  diss)
                        pos.append( [im[i] ,iR[j] ,jm[i] ,jR[j]] )
                    if diss<1.1:
                        connected =True
            if not connected:
                idis=np.where (dis ==np.min(dis))
                for i in range(len(dis)):
                    P=pos[i]
                    p1i=P[0]; p2i=P[1];p1j=P[2];p2j=P[3]
                    I1=np.min([p1i,p2i])
                    I2=np.max([p1i,p2i])
                    J1=np.min([p1j,p2j])
                    J2=np.max([p1j,p2j])
                    for i1 in range(I1,I2):
                        for j1 in range(J1,J2):
                            near_RV = RV1[i1-2:i1+2 , j1-2:j1+2]
                            near_Myo = Myo1[i1-2:i1+2 , j1-2:j1+2]
                            
                            if np.sum(near_Myo)>0.5*np.sum(near_RV):
                                RV1[i1 , j1]=0
                                Myo1[i1 , j1]=np.max(Myo1)
                            else:
                                RV1[i1 , j1]=np.max(RV1)
                                Myo1[i1 , j1]=0
                    




        # # # #   STEP 1 : myo extraction 


        # dialated LV1 with mean  radius of myo
        k=3
        # k=int(mean_radius)

        if np.sum(LV1) >0:
            kernel = np.ones((k, k), np.uint8)
            LV12 = cv2.dilate(LV1, kernel, iterations=1)
            Myo2 = LV12-LV1
            iii,jjj = np.where (Myo2>0)
            for i in range(len(iii)):
                Myo1[iii[i] , jjj[i]]=255

        #  Try con 3 :in myo we have BG (so subscrub LV from myo)
        #  Try con0  maximum partion
        # Try con 3 :in myo we have BG (so subscrub LV from myo) 

        
        image = copy.deepcopy(LV1)
        import imutils
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        LV1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(Myo1);cv2.imwrite('temp.png', image);image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        Myo1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(RV1)
        RV=0*RV
        cv2.imwrite('temp.png', image);image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        RV1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(BG);cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        BG1=copy.deepcopy(image[:,:,0])


        il,jl = np.where (LV1);iR,jR = np.where (RV1);im,jm = np.where (Myo1)
        for i in range(len (il)):
            for j in range(len (Myo1)):
                Myo1[ il[i] , jl[i]  ]=0

        if np.sum(LV1) >0 and np.sum(Myo1) >0  :
            kernel = np.ones((k, k), np.uint8)
            LV12 = cv2.dilate(LV1, kernel, iterations=1)
            Myo2 = LV12-LV1
            iii,jjj = np.where (Myo2>0)
            for i in range(len(iii)):
                Myo1[iii[i] , jjj[i]]=np.max(Myo1)


        # ########## step 2 myo extraction

        #  find mean radius of myo to fill regiron on myo is connected to BG
        _, binary_image = cv2.threshold(Myo1+LV1, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_radius = 0
        num_contours = 0

        for contour in contours:
            (x, y), radius_m = cv2.minEnclosingCircle(contour)

        _, binary_image = cv2.threshold(LV1, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y), radius_l = cv2.minEnclosingCircle(contour)
        mean_radius= radius_m-radius_l
        if show>0:
            print("Mean Radius:", mean_radius)

        # # # #   STEP 1 : myo extraction 


        # dialated LV1 with mean  radius of myo
        # k=3
        k=int(mean_radius+3)

        if np.sum(LV1) >0:
            kernel = np.ones((k, k), np.uint8)
            LV12 = cv2.dilate(LV1, kernel, iterations=1)
            Myo2 = LV12-LV1
            iii,jjj = np.where (Myo2>0)
            for i in range(len(iii)):
                Myo1[iii[i] , jjj[i]]=255

        #  Try con 3 :in myo we have BG (so subscrub LV from myo)
        #  Try con0  maximum partion
        # Try con 3 :in myo we have BG (so subscrub LV from myo) 

        
        image = copy.deepcopy(LV1)
        import imutils
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        LV1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(Myo1);cv2.imwrite('temp.png', image);image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        Myo1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(RV1)
        RV=0*RV
        cv2.imwrite('temp.png', image);image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        RV1=copy.deepcopy(image[:,:,0])

        image = copy.deepcopy(BG);cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png');os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1;max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area;        max_contour = contour
        cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1);i1=np.where (image<100);image[i1]=0;
        BG1=copy.deepcopy(image[:,:,0])


        il,jl = np.where (LV1);iR,jR = np.where (RV1);im,jm = np.where (Myo1)
        for i in range(len (il)):
            for j in range(len (Myo1)):
                Myo1[ il[i] , jl[i]  ]=0

        if np.sum(LV1) >0 and np.sum(Myo1) >0  :
            kernel = np.ones((k, k), np.uint8)
            LV12 = cv2.dilate(LV1, kernel, iterations=1)
            Myo2 = LV12-LV1
            iii,jjj = np.where (Myo2>0)
            for i in range(len(iii)):
                Myo1[iii[i] , jjj[i]]=np.max(Myo1)


        # # # #  remove region have Myo and RV  ==> just Myo
        iii1 = np.where (Myo1>0)
        anti_Myo1= 0*Myo1+1
        anti_Myo1[iii1]=0
        RV1=RV1*anti_Myo1

        # # # #  remove region have Myo and LV  ==> just LV
        iii1 = np.where (LV1>0)
        anti_LV1= 0*LV1+1
        anti_LV1[iii1]=0
        Myo1=Myo1*anti_LV1 

        # 
        BG1=0*BG1+255
        BG1=BG1-RV1
        BG1=BG1-Myo1
        BG1=BG1-LV1
        if show>0:
            plt.figure(2)
            plt.subplot(245);plt.imshow( (RV1) );plt.grid(False)
            plt.subplot(246);plt.imshow( (Myo1) );plt.grid(False)
            plt.subplot(247);plt.imshow( (LV1) );plt.grid(False)
            plt.subplot(248);plt.imshow( (BG1) );plt.grid(False)

        image_r=0*image3


        image_r[:,:,0] = RV1
        image_r[:,:,1] = Myo1
        image_r[:,:,2] = LV1


        if show>0:
            plt.figure(1);plt.subplot(131);plt.imshow(image3)    ;plt.title('predicted image') 
            plt.grid(False)

            plt.figure(1);plt.subplot(132);plt.imshow( image_r)   ;plt.title('Post processing image') ;
            plt.grid(False)
            plt.figure(1);plt.subplot(133);plt.imshow( np.abs (255*image3 - image_r)    );plt.title('Diff image') 
            plt.grid(False)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

            plt.pause(5);
            try:
                plt.savefig ( 'post_'+image_path)
            except:
                plt.savefig ( 'post_image.png' )
            
        image_r=image_r/255.0
    if not (np.sum(index_RV)>0  and np.sum(index_Myo)>0  and np.sum(index_LV)>0) :
        image_r=image3
    # except:image_r=image3    
    return image_r 




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




import numpy as np


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
T=''
def dice_function (Y,YY1):           
    d_r=1; d_l=1;      d_m=1;d_b=1;

    BG_p=0*YY1[:,:,0]+255;BG_p=BG_p-YY1[:,:,0];BG_p=BG_p-YY1[:,:,1]; BG_p=BG_p-YY1[:,:,2]
    BG_t=0*Y[:,:,0]+255;BG_t=BG_t-Y[:,:,0];BG_t=BG_t-Y[:,:,1]; BG_t=BG_t-Y[:,:,2]
    

    from metricss import dice_coef
    d_r=0.0001*round((np.mean(10000*dice_coef(Y[:,:,0], YY1[:,:,0], smooth=1))),2)
    d_m=0.0001*round((np.mean(10000*dice_coef(Y[:,:,1], YY1[:,:,1], smooth=1))),2)
    d_l=0.0001*round((np.mean(10000*dice_coef(Y[:,:,2], YY1[:,:,2], smooth=1))),2)
    d_b=0.0001*round((np.mean(10000*dice_coef(BG_t, BG_p, smooth=1))),2)

    d_overal = ( (d_l+d_r+d_m+d_b))/4
    d_overal  = round( d_overal , 2)
    
    d_r = round (d_r,4)
    d_m = round (d_m,4)
    d_l = round (d_l,4)
    d_overal = round (d_overal,4)
    return d_overal,d_r,d_l,d_b,d_m 






example=False
# example=True 
if example :
    import numpy as np
    import cv2 
    import matplotlib.pyplot as plt
    import os 
    plt.close ('all')
    os.startfile(os.getcwd())
    image_path ='5.png'
    image= cv2.imread(image_path)
    try:
        image=image[:,:,0]
    except:s=1
    a=np.unique(image)
    a=a[1:]
    image3=np.zeros([np.shape(image)[0] ,np.shape(image)[1] ,3 ])
    cnt=-1
    for i in a:
        i1,j1=np.where (image==i)
        cnt=cnt+1
        image3[i1,j1,cnt] = 1

    RV  = image3[:,:,0]
    Myo = image3[:,:,1]
    LV  = image3[:,:,2]
    BG   = 1-np.sum(image3,2)
    index_RV=np.where  (RV>0)
    index_Myo=np.where (Myo>0)
    index_LV=np.where  (LV>0)
    image_r =post_processing (image3,show=1)
    import matplotlib.pyplot as plt
    plt.figure(3000)
    plt.imshow(image_r)
    