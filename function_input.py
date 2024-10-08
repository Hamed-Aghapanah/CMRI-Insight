# =============================================================================
# function input/output    
# =============================================================================

def crop_image(I):
    import cv2 
    import numpy as np 
    cv2.imwrite('image_befor_crop.jpg',I)
    print('---------------------------------------------------')
    print('Please Click on imageg') 
    print('Then Press below keyes : ') 
    print('S (3 times)  for Save1 a (3 times) for Save2   ESP for exit') 
    print('---------------------------------------------------')
    cv2.imshow('image_befor_crop.jpg',I)
    def detect_quadrant(event, x, y, flags, param): 
        if event == cv2.EVENT_LBUTTONDOWN: 
            global xx,yy,second,image01,x0,y0,y1,x1
            xx=x
            yy=y
            print('image0 = cv2.imread(image_befor_crop.jpg)')
            image0 = cv2.imread('image_befor_crop.jpg')
            image000=image0.copy
            if second <=1:
                x0=x
                y0=y
                image01=image0[0:y,0:x,0:3]
                image00=image0
                cv2.line(image00,(x,0),(x,y),(255,20,255),5)
                cv2.line(image00,(0,y),(x,y),(0,255,255),5)
                height, width = image00.shape[:2]
                cv2.line(image00,(x,height),(x,y),(255,255,0),5)
                cv2.line(image00,(width,y),(x,y),(100,105,155),5)
                cv2.waitKey(50),cv2.imshow('first click, pess s then a', image00)
                cv2.waitKey(3000),
                c = cv2.waitKey(1) 
                if c == ord('s'):
                    second=2
                    
            if second ==2:
                x_min=np.min([x,x0])
                y_min=np.min([y,y0])
                x_max=np.max([x,x0])
                y_max=np.max([y,y0])
                x0=x_min
                y0=y_min
                x1=x_max
                y1=y_max
                img1111=image0
                cv2.imshow('first click, pess s then a', img)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font=cv2.FONT_HERSHEY_TRIPLEX
                height, width = img.shape[:2]
                            
                cv2.putText(img1111,'Croped image',(round(height/2)-30,round(width/2)-30), font, 1,(255,255,255),1,cv2.LINE_AA)
                cv2.line(img1111,(x0,y0),(x0,y1),(255,255,20),5)
                cv2.waitKey(50),cv2.imshow('first click, pess s then a', img1111) 
                cv2.line(img1111,(x1,y0),(x1,y1),(255,20,255),5)
                cv2.waitKey(50),cv2.imshow('first click, pess s then a', img1111) 
                cv2.line(img1111,(x0,y0),(x1,y0),(20,255,255),5)
                cv2.waitKey(50) ,cv2.imshow('first click, pess s then a', img1111)
                cv2.line(img1111,(x0,y1),(x1,y1),(233,255,213),5)
                cv2.waitKey(50) 
                cv2.imshow('first click, pess s then a', img1111)
                c = cv2.waitKey(1) 
                if c == ord('s'):
                    second=3
                if c == ord('z'):
                    second=1
     
    if __name__=='__main__': 
        width, height = 640, 480 
        img = 255 * np.ones((height, width, 3), dtype=np.uint8) 
        cv2.namedWindow('first click, pess s then a') 
        cv2.setMouseCallback('first click, pess s then a', detect_quadrant, {"img": img})
    #    print('x = ',x,'  y=',y)
        cnt=0
        x0=0,
        y0=0,
        y1=0,
        second=0
        image0 = cv2.imread('image0.jpg')
        #        print('x = ',x,'  y=',y)
        height, width = image0.shape[:2]
        x1=height,
        y1=width,
        second=0
        while True: 
            cnt=cnt+1
            if second <=1:
                image0 = cv2.imread('image0.jpg')
                cv2.imshow('first click, pess s then a', image0)
                
                height, width = image0.shape[:2]
            c = cv2.waitKey(1) 
            if second ==2:
                c = cv2.waitKey(1) 
                if c == ord('a'):
                    second=3
                    print('crop is done',second)
                    image0100=image0[y0:y1,x0:x1,:]
                    cv2.imshow('croped image',image0100 )
                    cv2.imwrite('image01.jpg',image0100)
                    cv2.waitKey(3000) 
                    break
                    
            if c == ord('s'):
                second=2
           
            if c == 27: 
                break 
        cv2.destroyAllWindows()
        return ( image0100 ) 



# =============================================================================
# camera
# =============================================================================
def camera(s):
    capturing=1
    cur_mode0=0
    if capturing==1:
        import cv2 as cv
        cap1 = cv.VideoCapture(-2)

        cap = cv.VideoCapture(0)
        ok=1
        cur_mode = None
        print('Please Press ')
        print('S  for save ,  Z for undo      ESC for exit')
        while(ok):
        # Take each frame
            _, image0 = cap.read()
            output =image0
            c = cv.waitKey(1)
            cv.imshow('please press :S  for save ,  Z for undo ESC for exit (gyflxcuvhba)',output)
            if c == 27: 
                break 
            if c != -1 and c != 255 and c != cur_mode:
                cur_mode = c 
            if cur_mode == ord('g'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2GRAY) 
            elif cur_mode == ord('y'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2YUV) 
            elif cur_mode == ord('f'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2HLS_FULL) 
            elif cur_mode == ord('l'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2LAB) 
            elif cur_mode == ord('x'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2XYZ) 
            elif cur_mode == ord('c'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2YCrCb) 
            elif cur_mode == ord('u'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2Luv) 
            elif cur_mode == ord('v'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2LUV) 
            elif cur_mode == ord('h'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2HSV) 
            elif cur_mode == ord('b'): 
                output = cv.cvtColor(image0, cv.COLOR_BGR2RGB)
            elif cur_mode == ord('a'): 
                output = image0
            elif cur_mode == ord('s'):     
                cv.imwrite('image0.jpg',image0)
                print('image is saved. :)')
                cv.imshow('please press :S  for save ,  Z for undo ESC for exit (gyflxcuvhba)',image0)
                cur_mode = ord(']')
                cv.waitKey(2000)
    #            break
            elif cur_mode == ord('z'):     
                cv.imshow('please press :S  for save ,  Z for undo ESC for exit (gyflxcuvhba)',output)
                cur_mode = ord('p')
                cur_mode0 = ord('p')
                print('press again any key')
                
            elif cur_mode0 == ord('p'):
                if cur_mode == ord('s'):
                    break
                
            cv.imshow('please press :S  for save ,  Z for undo ESC for exit (gyflxcuvhba)',output)
        cap.release() 
        cv.destroyAllWindows()
    return(output)   

# =============================================================================
# save_image
# =============================================================================
def save_image(result):
    import cv2 as cv
    cv.imwrite('result.jpg',result)
    return () 