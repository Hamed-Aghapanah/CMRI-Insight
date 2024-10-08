# =============================================================================
# function segmentation   
# =============================================================================
def deep_mstganet_segmentation (I3 , mask1=3):
    import numpy as np
    from function_predictor import function_predictor1 
    if np.size(mask1)==1:
        mask1 =0*I3
    YY1 , p, superimposed_img1, superimposed_img =function_predictor1(I3,mask1)
    return  YY1 , p, superimposed_img1, superimposed_img






def entropy_segmentation (I3):
    import numpy as np
    import cv2
    from PIL import Image
    
    img=I3
    kernel = np.ones((15,15),np.uint8)
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 3)
    blur = cv2.blur(closing,(15,15))
    # Binarize the image
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Search for contours and select the biggest one
    contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    # Calculate the mean color of the contour
    mean = cv2.mean(res, mask = mask)
#    print(mean)
    
    # Make some sort of criterion as the ratio hair vs. skin color varies
    # thus makes it hard to unify the threshold.
    # NOTE that this is only for example and it will not work with all images!!!
#    th
#    
#    182
    if mean[2] >182:
        bp = mean[0]/100*35
        gp = mean[1]/100*35
        rp = mean[2]/100*35   
    
    elif 182 > mean[2] >160:
        bp = mean[0]/100*30
        gp = mean[1]/100*30
        rp = mean[2]/100*30
    
    elif 160>mean[2]>150:
        bp = mean[0]/100*50
        gp = mean[1]/100*50
        rp = mean[2]/100*50
    
    elif 150>mean[2]>120:
        bp = mean[0]/100*60
        gp = mean[1]/100*60
        rp = mean[2]/100*60
    
    else:
        bp = mean[0]/100*53
        gp = mean[1]/100*53
        rp = mean[2]/100*53
    
    # Write temporary image
    cv2.imwrite('temp.png', res)
    
    # Open the image with PIL and load it to RGB pixelpoints
    mask2 = Image.open('temp.png')
    pix = mask2.load()
    x,y = mask2.size
    
    # Itearate through the image and make some sort of logic to replace the pixels that
    # differs from the mean of the image
    # NOTE that this alghorithm is for example and it will not work with other images
    
    for i in range(0,x):
        for j in range(0,y):
            if -1<pix[i,j][0]<bp or -1<pix[i,j][1]<gp or -1<pix[i,j][2]<rp:
                try:
                    pix[i,j] = b,g,r
                except:
                    pix[i,j] = (int(mean[0]),int(mean[1]),int(mean[2]))
            else:
                b,g,r = pix[i,j]
    
    
           
    res = np.array(mask2)
    res = res[:,:,::-1].copy()
    final = cv2.bitwise_and(res, res, mask=mask)

    
    sssss=img 

    
    sssss[:,:,0]=(mask/np.max(mask))*img[:,:,0] 
    sssss[:,:,1]=(mask/np.max(mask))*img[:,:,1]
    sssss[:,:,2]=(mask/np.max(mask))*img[:,:,2]


    return ( sssss ,mask)    

# =============================================================================
# thershold_segmentation
# =============================================================================
def thershold_segmentation (I3,th):
#    def thershold_segmentation (I3,th):
    import numpy as np
    import cv2
    from PIL import Image
    
   
    img=I3
    print('th',th)
    kernel = np.ones((th,th),np.uint8)
    # Perform closing to remove hair and blur the image
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)
    blur = cv2.blur(closing,(th,th))

    # Binarize the image
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+th)
#    _, thresh = cv2.threshold(gray,0,255,th+cv2.THRESH_OTSU)
    #cv2.imshow('thresh', thresh)
    
    # Search for contours and select the biggest one
    contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    
    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    # Calculate the mean color of the contour
    mean = cv2.mean(res, mask = mask)

    if mean[2] >182:
        bp = mean[0]/100*35
        gp = mean[1]/100*35
        rp = mean[2]/100*35   
    
    elif 182 > mean[2] >160:
        bp = mean[0]/100*30
        gp = mean[1]/100*30
        rp = mean[2]/100*30
    
    elif 160>mean[2]>150:
        bp = mean[0]/100*50
        gp = mean[1]/100*50
        rp = mean[2]/100*50
    
    elif 150>mean[2]>120:
        bp = mean[0]/100*60
        gp = mean[1]/100*60
        rp = mean[2]/100*60
    
    else:
        bp = mean[0]/100*53
        gp = mean[1]/100*53
        rp = mean[2]/100*53
    
    # Write temporary image
    cv2.imwrite('temp.png', res)
    
    # Open the image with PIL and load it to RGB pixelpoints
    mask2 = Image.open('temp.png')
    pix = mask2.load()
    x,y = mask2.size
    
    # Itearate through the image and make some sort of logic to replace the pixels that
    # differs from the mean of the image
    # NOTE that this alghorithm is for example and it will not work with other images
    
    for i in range(0,x):
        for j in range(0,y):
            if -1<pix[i,j][0]<bp or -1<pix[i,j][1]<gp or -1<pix[i,j][2]<rp:
                try:
                    pix[i,j] = b,g,r
                except:
                    pix[i,j] = (int(mean[0]),int(mean[1]),int(mean[2]))
            else:
                b,g,r = pix[i,j]
    
    
    #s,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # Transform the image back to cv2 format and mask the result         
    res = np.array(mask2)
    res = res[:,:,::-1].copy()
    final = cv2.bitwise_and(res, res, mask=mask)
#    cv2.imshow('real image', img)
#    cv2.imshow('res', mask)
    
    sssss=img 
    #mask=np.float16(mask)
    #mask=(mask-mask.min)/(mask.max  - mask.min)
    #mask=np.uint8(mask)
    
    sssss[:,:,0]=(mask/np.max(mask))*img[:,:,0] 
    sssss[:,:,1]=(mask/np.max(mask))*img[:,:,1]
    sssss[:,:,2]=(mask/np.max(mask))*img[:,:,2]
#    cv2.imshow('masked_image', sssss)
#    cv2.imwrite('image1_2.jpg',sssss)
    #cv2.imwrite('mask.jpg',mask)
    
    #sssss[:,:,0]=mask*img[:,:,0]
    #sssss[:,:,1]=mask*img[:,:,1]
    #sssss[:,:,2]=mask*img[:,:,2]
    #cv2.imshow('sssss image', sssss)
    # Display the result
    
    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return(sssss,mask2)
        

 

