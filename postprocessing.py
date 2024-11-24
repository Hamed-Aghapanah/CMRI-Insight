import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
from pathlib import Path

global Mask_input

def smoother(mask):
    smoothed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    return mask

def smoothing(a, pixels):
    import copy
    c = copy.deepcopy(a)
    for i in range(2, np.shape(a)[0] - 2):
        for j in range(2, np.shape(a)[1] - 2):
            if a[i, j] == 0:
                if np.sum(a[i - 1:i + 1, j - 1:j + 1]) >= pixels:
                    c[i, j] = 1
    return c




def connectivity_loss_base(Mask_input, method=1, show=0):
    Mask_input = np.float32(Mask_input)

    def modd(a):
        a = np.nan_to_num(a)
        a = np.float32(a)

    def meann(a):
        a = np.nan_to_num(a)
        a = np.float32(np.mean(a))
        if np.mean(a) > 0.05:
            return 1
        if np.mean(a) <= 0.05:
            return 0

    image_r = Mask_input
    for k in range(0, 3):
        for i in range(2, 127):
            for j in range(2, 127):
                r1 = Mask_input[i-1:i+2, j-1, k]
                r2 = Mask_input[i-1:i+2, j, k]
                r3 = Mask_input[i-1:i+2, j+1, k]
                
                r4 = Mask_input[i-1, j-1:j+2, k]
                r5 = Mask_input[i, j-1:j+2, k]
                r6 = Mask_input[i+1, j-1:j+2, k]
                
                r7 = [Mask_input[i-1, j-1, k], Mask_input[i, j, k], Mask_input[i+1, j+1, k]]
                r8 = [Mask_input[i+1, j-1, k], Mask_input[i, j, k], Mask_input[i-1, j+1, k]]
                
                r9 = Mask_input[i, j, k]

                r7 = np.array(r7)
                r8 = np.array(r8)

                r1 = r1.ravel()
                r2 = r2.ravel()
                r3 = r3.ravel()
                r4 = r4.ravel()
                r5 = r5.ravel()
                r6 = r6.ravel()
                r7 = r7.ravel()
                r8 = r8.ravel()
                r9 = r9.ravel()

                r_con = np.concatenate((r1, r2, r3, r4, r5, r6, r7, r8, r9))
                a = 0
                if method == 1:
                    a = modd(r_con)
                    image_r[i, j, k] = a
                if method == 2:
                    a = np.float32(np.mean(r_con))
                    image_r[i, j, k] = a

                if method == 3:
                    a = modd([modd(r1), modd(r2), modd(r3), modd(r4), modd(r5),
                              modd(r6), modd(r7), modd(r8), modd(r9)])
                    image_r[i, j, k] = a
                if method == 4:
                    a = meann([modd(r1), modd(r2), modd(r3), modd(r4), modd(r5),
                              modd(r6), modd(r7), modd(r8), modd(r9)])
                    image_r[i, j, k] = a
                if method == 5:
                    a = modd([meann(r1), meann(r2), meann(r3), meann(r4), meann(r5),
                              meann(r6), meann(r7), meann(r8), meann(r9)])
                    image_r[i, j, k] = a
                if method == 6:
                    a = meann([meann(r1), meann(r2), meann(r3), meann(r4), meann(r5),
                              meann(r6), meann(r7), meann(r8), meann(r9)])
                    image_r[i, j, k] = a
                if method == 7:
                    a = modd(r2)
                    image_r[i, j, k] = a
                if method == 8:
                    a = modd(r5)
                    image_r[i, j, k] = a
                if method == 9:
                    a = modd([modd(r2), modd(r5)])
                    image_r[i, j, k] = a
                if method == 10:
                    a = modd(r9)
                    image_r[i, j, k] = a
        image_r = np.nan_to_num(image_r)
    return image_r


def post_processing(Mask_input):
    import numpy as np
    RV = Mask_input[:, :, 0]
    Myo = Mask_input[:, :, 1]
    LV = Mask_input[:, :, 2]
    
    if np.sum(RV) == 0 or np.sum(Myo) == 0 or np.sum(LV) == 0:
        return None, Mask_input
    
    LV = smoother(LV)
    RV = smoother(RV)
    Myo = smoother(Myo)
    
    BG = 1 - np.sum(Mask_input, 2)
    index_RV = np.where(RV > 0)
    index_Myo = np.where(Myo > 0)
    index_LV = np.where(LV > 0)

    if np.sum(index_RV) > 0 and np.sum(index_Myo) > 0 and np.sum(index_LV) > 0:
        image = copy.deepcopy(LV)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png')
        os.remove('temp.png')
         
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        try:
            cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1)
            i1 = np.where(image < 100)
            image[i1] = 0
        except:
            pass
        LV1 = copy.deepcopy(image[:, :, 0])
        
        
        s_LV1 =np.sum(LV1)/np.max(LV1)
        s_Myo=np.sum(Myo)/np.max(Myo)
        
        a= int(np.fix(2*s_Myo/360))
        # print( a) 

        kernel = np.ones((a, a), np.uint8)
        
        image = copy.deepcopy(Myo+ smoothing (LV1,a) +   cv2.dilate(LV1, kernel, iterations=1) )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png')
        os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        try:
            cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1)
            i1 = np.where(image < 100)
            image[i1] = 0
        except:
            pass
        Myo1 = copy.deepcopy(image[:, :, 0])
        
        
        
        
        # plt.imshow(Myo1)
        # aa

        image = copy.deepcopy(RV)
        RV = 0 * RV
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png')
        os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        try:
            cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1)
            i1 = np.where(image < 100)
            image[i1] = 0
        except:
            pass
        RV1 = copy.deepcopy(image[:, :, 0])
        
        
        
        
        
        
        image = copy.deepcopy(BG)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        cv2.imwrite('temp.png', image)
        image = cv2.imread('temp.png')
        os.remove('temp.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        try:
            cv2.drawContours(image, [max_contour], 0, (255, 255, 255), -1)
            i1 = np.where(image < 100)
            image[i1] = 0
        except:
            pass
        
        BG1 = copy.deepcopy(image[:, :, 0])

      
        
        import numpy as np
        from scipy.spatial import cKDTree
        
        il, jl = np.where(LV1)
        iR, jR = np.where(RV1)
        
        # Combine iR and jR into a single array of coordinates
        RV_coords = np.column_stack((iR, jR))
        
        # Create a k-d tree for fast nearest-neighbor search
        tree = cKDTree(RV_coords)
        
        # Combine il and jl into a single array of coordinates
        LV_coords = np.column_stack((il, jl))
        
        # Query the k-d tree to find the distances and indices of the nearest neighbors
        distances, indices = tree.query(LV_coords, distance_upper_bound=5)
        
        # Filter out the points where the distance is greater than 2
        within_distance = distances < 5
        
        # Apply the conditions to LV1 and Myo1
        LV1[il[within_distance], jl[within_distance]] = 0
        Myo1[il[within_distance], jl[within_distance]] = np.max(Myo1)
        
        

        
        
        
        import numpy as np
        from numba import njit, prange
        
        @njit(parallel=True)
        def process_image(image, max_value):
            rows, cols = image.shape
            for i in prange(3, rows - 3):
                for j in range(3, cols - 3):
                    if np.mean(image[i - 2:i + 2, j - 2:j + 2]) > 0.8 * max_value:
                        image[i, j] = max_value
            return image
        
        def process_all_images(Myo1, LV1, RV1):
            if np.sum(Myo1) > 0:
                max_val_myo = np.max(Myo1)
                Myo1 = process_image(Myo1, max_val_myo)
        
            if np.sum(LV1) > 0:
                max_val_lv = np.max(LV1)
                LV1 = process_image(LV1, max_val_lv)
        
            if np.sum(RV1) > 0:
                max_val_rv = np.max(RV1)
                RV1 = process_image(RV1, max_val_rv)
        
            return Myo1, LV1, RV1
        
        # نمونه استفاده:
        Myo1, LV1, RV1 = process_all_images(Myo1, LV1, RV1)
        
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_red = cv2.dilate(RV1, kernel, iterations=3)
        dilated_green = cv2.dilate(LV1, kernel, iterations=3)

        # عملیات closing برای پر کردن حفره‌های کوچک در مرزها
        closed_red = cv2.morphologyEx(dilated_red, cv2.MORPH_CLOSE, kernel)
        closed_green = cv2.morphologyEx(dilated_green, cv2.MORPH_CLOSE, kernel)

        # پیدا کردن ناحیه‌های مشترک بین ناحیه‌های قرمز و سبز گسترش یافته
        border_area = cv2.bitwise_and(closed_red, closed_green)

        # رنگ‌آمیزی ناحیه‌های مشکی بین نواحی قرمز و سبز به رنگ قرمز
        # image[border_area == 255] = [0, 0, 255]
        
        RV1 [border_area > 0] = np.max(RV1)
        # Myo1 [border_area > 0] = np.max(Myo1)

        # maskRV1 = RV1 > 0
        maskLV1 = LV1 > 0
        Myo1[maskLV1] = 0
        
        maskMyo1 = Myo1 > 0
        RV1[maskMyo1] = 0
        RV1[maskLV1] = 0

       

        BG1 = 0 * BG1 + 255
        BG1 = BG1 - RV1
        BG1 = BG1 - Myo1
        BG1 = BG1 - LV1
        
        image_r = np.zeros_like(Mask_input)
        # image_r[:, :, 0] = BG
        image_r[:, :, 0] = RV1
        image_r[:, :, 1] = Myo1
        image_r[:, :, 2] = LV1
        image_r1=image_r
        
        # image_r = connectivity_loss_base (image_r)
        
        
        image_r1 = image_r1 / 255.0

        image_r = image_r / 255.0
    else:
        image_r = Mask_input
        image_r1= Mask_input

    return image_r,image_r1

if __name__ == "__main__":
    for i in range(1,10):
    # for i in range(8,9):
        plt.close('all')
        image_path = 'phantom ('+str(i)+').png'
        image = cv2.imread(image_path)/255
        
        num_channels=3        
        RV = image[:, :, 0]
        Myo = image[:, :, 1]
        LV = image[:, :, 2] if num_channels > 2 else np.zeros_like(RV)  # Handle case where less than 3 channels exist
        BG = 1 - np.sum(image, 2)
        index_RV = np.where(RV > 0)
        index_Myo = np.where(Myo > 0)
        index_LV = np.where(LV > 0)
        
    
    
        image_r = np.zeros([np.shape(image)[0], np.shape(image)[1], 3])  # 4 channels
        
        # Assign values to image_r
        image_r[:, :, 0] = RV
        image_r[:, :, 1] = Myo
        image_r[:, :, 2] = LV
        # image_r[:, :, 3] = BG
       
        
        import time
        start_time = time.time()
        image_r ,image_r1= post_processing(image_r)
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
                                                                          
        show=True
        if show  :
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8) # Now convert RGB to BGR image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image_r = cv2.normalize(image_r, None, 0, 255, cv2.NORM_MINMAX)
            image_r = image_r.astype(np.uint8) # Now convert RGB to BGR image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_r_bgr = cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR)
            plt.figure(1)
            plt.subplot(131)
            plt.imshow(image_bgr)
            plt.title('first mask '+image_path)
            plt.grid(False)
    
            plt.figure(1)
            plt.subplot(132)
            plt.imshow(image_r_bgr)
            plt.title('Post Processing mask')
            plt.grid(False)
    
            plt.figure(1)
            plt.subplot(133)
            plt.imshow(np.abs( image_bgr - image_r_bgr))
            plt.title('Difference masks')
            plt.grid(False)
            
         
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
    
            plt.pause(1.5)
            try:
                plt.savefig('post_' + image_path)
            except:
                plt.savefig('post_image.png')
                


