"""   Created on Fri Jan 19 02:09:26 2024

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""
# =============================================================================
# import libs
# =============================================================================
import cv2
import numpy as np
import copy 
import matplotlib.pyplot as plt
import pydicom
import os 
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
cv2.destroyAllWindows()

# =============================================================================
# functions
# =============================================================================

# def store_evolution_in(lst):
#     """Returns a callback function to store the evolution of the level sets in
#     the given list.
#     """

#     def _store(x):
#         lst.append(np.copy(x))

#     return _store
global image0

brightness = 0  # مقدار اولیه روشنایی


def manual_segmentator(mask00, image0 ,color1,color2 ,region ,frame, dicom_data1, npz_name,):
    # Resize the image
    import numpy as np
    
    # image00=copy.deepcopy(image)
   
    try:
        image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
    except:print(1)
    image_with_points = image0.copy()
    image = image0.copy()
    image002 = image.copy()
    # Create a window for the image
    wname='please select ' +region+' in frame '+str(frame)
    cv2.namedWindow(wname)
    
    # List to store selected points
    points = []  

         
    def adjust_brightness(image, brightness):
        """
        تغییر روشنایی تصویر و اعمال هیستوگرام equalization
        """
        # تغییر روشنایی با اضافه کردن مقدار ثابت
        new_image = np.clip(image + brightness, 0, 255).astype(np.uint8)
        
        # اعمال هیستوگرام equalization فقط روی کانال روشنایی (برای تصاویر رنگی)
        if len(new_image.shape) == 3:  # اگر تصویر رنگی است
            yuv = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # هیستوگرام فقط روی کانال Y
            new_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:  # اگر تصویر خاکستری است
            new_image = cv2.equalizeHist(new_image)
        
        return new_image

    brightness = 0    
    def click_event(event, x, y, flags, param):
        global brightness
        color1 = (0, 255, 255)
        
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            points.append((x, y))
            image = image0.copy()
            for pt in points:  # Redraw all remaining points
                cv2.circle(image, pt, 2, color1, -1)
            # cv2.circle(image, (x, y), 2, color1, -1)
            wname='please select ' +region+' in frame '+str(frame)
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
            
            cv2.imshow(wname, image)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button click
            if points: 
                points.pop()
                image = image0.copy()
                for pt in points:  # Redraw all remaining points
                    cv2.circle(image, pt, 2, color1, -1)
                wname='please select ' +region+' in frame '+str(frame)
                cv2.imshow(wname, image)
                
        elif event == cv2.EVENT_MOUSEWHEEL:  # Mouse wheel scroll
            if flags > 0:  # Scroll up
                brightness += 10  # افزایش روشنایی
            else:  # Scroll down
                brightness -= 10  # کاهش روشنایی
            
            # اعمال تغییر روشنایی
            adjusted_image = adjust_brightness(image0, brightness)
            image = adjusted_image.copy()
            
            # دوباره رسم نقاط روی تصویر تغییر یافته
            for pt in points:
                cv2.circle(image, pt, 2, color1, -1)
            
            wname = 'please select ' + region + ' in frame ' + str(frame)
            cv2.imshow(wname, image)
            
            
            
    cv2.setMouseCallback(wname, click_event)
    # Display the image
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    # cv2.moveWindow(wname, 20, 20)  
    cv2.imshow(wname, image)
    
    
    cv2.waitKey(0)
    
    # Close the image window
    cv2.destroyAllWindows()
    points_array = np.array(points)
    
    
    # Create an empty mask to draw the contour
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # cv2.drawContours(mask, [interp_curve], 0, (0, 0, 255), thickness=cv2.FILLED)
    cv2.drawContours(mask, [points_array], 0, color1, thickness=cv2.FILLED)
    
    print(np.shape(points_array))
    # Add the mask to the original image
    result = cv2.addWeighted(image, 1, mask, 0.5, 0)
    
    
    window_size=1;threshold=1
    # mask = region_growing(image, mask, threshold,window_size)
    mask002 = cv2.GaussianBlur(mask, (1, 1), 0)
    _, mask002 = cv2.threshold(mask002, 0.0001, 255, cv2.THRESH_BINARY)
    try:mask003 = cv2.cvtColor(mask002, cv2.COLOR_BGR2GRAY)
    except:mask003=mask002   
    _, mask003= cv2.threshold(mask003, 0.0001, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask003, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points_section1 = max(contours, key=cv2.contourArea)
    cv2.drawContours(result, [points_section1], 0, color1, 1)
    
    # cv2.waitKey(1)
    # print('region growing')
    mask_out = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_out = cv2.normalize(mask_out, None, 0, 255, cv2.NORM_MINMAX)
    mask_out = mask_out.astype(np.uint8)
    
    segmentation_mask =mask_out-mask00
   
    print(np.unique(segmentation_mask))
    _, segmentation_mask = cv2.threshold(segmentation_mask, 1, 255, cv2.THRESH_BINARY)
    
    smoothed_segmentation_result = cv2.GaussianBlur(segmentation_mask, (1, 1), 0)
    _, segmentation_result3 = cv2.threshold(smoothed_segmentation_result, 0.0001, 255, cv2.THRESH_BINARY)  
    contours, _ = cv2.findContours(segmentation_result3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     # Find contours
    points_section1 = max(contours, key=cv2.contourArea)
# Iterate over the contour points and extract them
    point2 = []
    for point in points_section1:
        x, y = point[0]
        point2.append((x, y))
    print('np.shape(point2) = ',np.shape(point2)) 
    import numpy as np
    num_points_desired = 79
    num_points_point2 = len(point2)
    
    pointx =point2[0][0] +point2[-1][0] 
    pointy =point2[0][1] +point2[-1][1] 
    point2.append( (int(pointx/2) , int(pointy/2) ) )
    
    distances_point2 = [np.linalg.norm(np.array(point2[i]) - np.array(point2[i - 1])) for i in range(1, num_points_point2)]
    total_distance_point2 = sum(distances_point2)
    
    # محاسبه مقدار فاصله میان هر نقطه در point3
    interval_length_point2 = total_distance_point2 / (num_points_desired +1)
    
    # نقاط point3
    point3 = []
    
    # نقطه اول از point2
    point3.append(point2[0])
    current_distance = 0
    
    i = 1
    
    while len(point3) < num_points_desired and i < num_points_point2:
        while current_distance + distances_point2[i - 1] >= interval_length_point2 * len(point3):
            ratio = (interval_length_point2 * len(point3) - current_distance) / distances_point2[i - 1]
            new_point = (
                int((1 - ratio) * point2[i - 1][0] + ratio * point2[i][0]),
                int((1 - ratio) * point2[i - 1][1] + ratio * point2[i][1])
            )
            point3.append(new_point)
        
        current_distance += distances_point2[i - 1]
        i =i +1
    if np.shape(point3)[0]<80:
        print(point3[0])
        pointx =point3[0][0] +point3[-1][0] 
        pointy =point3[0][1] +point3[-1][1] 
        point3.append( (int(pointx/2) , int(pointy/2) ) )
    # Now point3 contains 80 points resampled from point2 with the first point having maximum x and y coordinates
    point1 = np.squeeze(points_section1, axis=1)
    np.savez(npz_name+'.npz', point1=point1, dicom_data1=dicom_data1)
    print('np.shape(point1) = ',np.shape(point1))
    print('np.shape(point2) = ',np.shape(point2))
    print('np.shape(point3) = ',np.shape(point3))
       
    
    
    # Display the image with points# Draw circles at each point
    
    
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    return point2,point3,segmentation_mask
    


def segmentator (pixel_array,i):
    for region in ['RV' ,'LV','Myo']:
        # print('image ', i ,' th from ', np.shape(pixel_array)[0], ' ==> select '+region)
        # image0=pixel_array[i]
        if np.size (np.shape(pixel_array))==3:
            image000 =pixel_array[i] 
        else:
            image000 =pixel_array
        normalized_img = cv2.normalize(image000, None, 0, 255, cv2.NORM_MINMAX)
        uint8_img = normalized_img.astype(np.uint8)
        image000= uint8_img
        image0 = cv2.cvtColor(image000.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        frame=i
        dicom_data1=0
        npz_name='temp '+region+'_frame  '+str(frame)
        if 'R' in region :
            result=image0
        new_width = 640
        # new_height = 650
        shapee = np.shape (image0) 
        aspect_ratio = shapee[1] / shapee[0]
        new_height = int(new_width * aspect_ratio)
            
        image = cv2.resize(image0, (new_width, new_height))
        # import deepcopy
        saved_image = image
        if 'R' in region :
            color1=(255, 0, 0)
            color2 =(255, 255, 0)
            mask_L=0*image
            mask_L= cv2.cvtColor(mask_L, cv2.COLOR_BGR2GRAY)
            mask00=mask_L
        if 'L' in region :
            color1=(0, 255,0 )
            color2 =(255, 0, 255)
        if 'M' in region :
            color1=(0, 0, 255)
            color3=(255, 0, 0)
            color2 =(0, 255, 255)
            
        point2,point3,mask_out = manual_segmentator(mask00, image ,color1,color2,region ,frame,dicom_data1, npz_name)
        thickness = 2
        if 'R' in region :
            mask_out3=0*image
            mask_out3[:,:,2]=mask_out
            result = cv2.addWeighted(image, 1, mask_out3, 0.5, 0)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            # cv2.moveWindow("Result", 200, 200)  
            cv2.imshow("Result", result)
            
            contour = np.array([point3], dtype=np.int32)
            for x,y in point3 :
                cv2.circle(result, (x, y), 1, color1, 1)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", result)
            mask_r = mask_out
            mask00 = mask_r
            mask_r3=mask_out3
            point_r=point3
        if 'L' in region :
            mask_out3=0*image
            mask_out3[:,:,0]=mask_out
            mask_l3=mask_out3
            result = cv2.addWeighted(image, 1, mask_out3, 0.5, 0)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            # cv2.moveWindow("Result", 200, 200)  
            cv2.imshow("Result", result)
            contour = np.array([point3], dtype=np.int32)
            for x,y in point3 :
                cv2.circle(result, (x, y), 1, color2, 1)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            # cv2.moveWindow("Result", 200, 200)  
            cv2.imshow("Result", result)
            mask_l = mask_out
            mask00 = mask_r+mask_l
            point_l=point3
        if 'M' in region :
            mask_out3=0*image
            mask_out3[:,:,1]=mask_out
            result = cv2.addWeighted(image, 1, mask_out3, 0.5, 0)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            # cv2.moveWindow("Result", 200, 200)  
            cv2.imshow("Result", result)
            contour = np.array([point3], dtype=np.int32)
            for x,y in point3 :
                cv2.circle(result, (x, y), 1, color3, 1)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            # cv2.moveWindow("Result", 200, 200)  
            cv2.imshow("Result", result)
            mask_m = mask_out
            mask_m3=mask_out3
            point_m=point3
            
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    result=cv2.addWeighted(image, 1, mask_r3, 0.5, 0)
    result=cv2.addWeighted(result, 1, mask_l3, 0.5, 0)
    result=cv2.addWeighted(result, 1, mask_m3, 0.5, 0)
    # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("Result", 200, 200)  
    # cv2.imshow("Result", result)    
    for x,y in point_r :
        cv2.circle(result, (x, y), 1, color1, 1)
    for x,y in point_l :
        cv2.circle(result, (x, y), 1, color2, 1)
    for x,y in point_m :
        cv2.circle(result, (x, y), 1, color3, 1)
    # cv2.namedWindow('Result of fram '+str(i), cv2.WINDOW_NORMAL) 
    # cv2.moveWindow('Result of fram '+str(i), 1000, 200)    
    cv2.imshow('Result of fram '+str(i), result)  
    
    return saved_image, mask_r  ,mask_l , mask_m ,point_r, point_l,  point_m

# =============================================================================
# read Dicom data
# =============================================================================
example =True
example =False


if example:
        
    path0=os.getcwd()
    
    region='L.V.'
    
    
    dicom_path = path0+"\\s.dcm"
    dicom_path=r'F:\0001phd\00_thesis\0_mfiles\1_local_dataset\01_dicom viewer\01_DICOM_viewer\1705364\DICOM\series0007-Body\img0001MultiFrame25-unknown.dcm'
    dicom_data = pydicom.dcmread(dicom_path)
    
    # Get the pixel data
    pixel_array = dicom_data.pixel_array
    
    print(np.shape(pixel_array))
    
    # Display the DICOM image
    
    if np.size( np.shape(pixel_array))==2:
        plt.imshow(pixel_array, cmap=plt.cm.gray)
        plt.title("DICOM Image")
        plt.show()
        image0=pixel_array
    
    
    
    if np.size( np.shape(pixel_array))==3:
        for i in range( np.shape(pixel_array)[0]):
            try:
                saved_image, mask_r  ,mask_l , mask_m ,point_r, point_l,  point_m =segmentator (pixel_array,i)
            except:pass
            
        




