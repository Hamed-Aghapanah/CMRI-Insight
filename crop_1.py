#def crop_1(image0):
import cv2 
import numpy as np 
def detect_quadrant(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN:       
        global xx,yy,second,image01,x0,y0,y1,x1
        xx=x
        yy=y
        image0 = cv2.imread('image01.jpg')
        image0  = cv2.resize(image0,(680,480))
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
            cv2.waitKey(50),cv2.imshow('First click, then pess 3 times s, last press a', image00)
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
            cv2.imshow('First click, then pess 3 times s, last press a', img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font=cv2.FONT_HERSHEY_TRIPLEX
            height, width = img.shape[:2]
                        
            cv2.putText(img1111,'Croped image',(round(height/2)-30,round(width/2)-30), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.line(img1111,(x0,y0),(x0,y1),(255,255,20),5)
            cv2.waitKey(50),cv2.imshow('First click, then pess 3 times s, last press a', img1111) 
            cv2.line(img1111,(x1,y0),(x1,y1),(255,20,255),5)
            cv2.waitKey(50),cv2.imshow('First click, then pess 3 times s, last press a', img1111) 
            cv2.line(img1111,(x0,y0),(x1,y0),(20,255,255),5)
            cv2.waitKey(50) ,cv2.imshow('First click, then pess 3 times s, last press a', img1111)
            cv2.line(img1111,(x0,y1),(x1,y1),(233,255,213),5)
            cv2.waitKey(50) 
            cv2.imshow('First click, then pess 3 times s, last press a', img1111)
            c = cv2.waitKey(1) 
            if c == ord('s'):
                second=3
            if c == ord('z'):
                second=1
 
if __name__=='__main__': 
    width, height = 640, 480 
    img = 255 * np.ones((height, width, 3), dtype=np.uint8) 
    cv2.namedWindow('First click, then pess 3 times s, last press a') 
    cv2.setMouseCallback('First click, then pess 3 times s, last press a', detect_quadrant, {"img": img})
    cnt=0
    x0=0,
    y0=0,
    y1=0,
    second=0
    image0 = cv2.imread('image0.jpg')
    image0  = cv2.resize(image0,(680,480))
    height, width = image0.shape[:2]
    x1=height,
    y1=width,
    second=0
    while True: 
        cnt=cnt+1
        if second <=1:
            cv2.imshow('First click, then pess 3 times s, last press a', image0)
            height, width = image0.shape[:2]
        c = cv2.waitKey(1) 
        if second ==2:
            c = cv2.waitKey(1) 
            if c == ord('a'):
                second=3
                print('crop is done',second)
                image0100=image0[y0:y1,x0:x1,:]
                image0100  = cv2.resize(image0100,(680,480))
                cv2.imshow('croped image',image0100 )
                cv2.imwrite('image01.jpg',image0100)
                cv2.waitKey(1000) 
                break
        if c == ord('s'):
            second=2
        if c == 27: 
            break 
    cv2.destroyWindow('croped image')
    cv2.destroyWindow('First click, then pess 3 times s, last press a')
    