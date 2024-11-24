# F:\papers\1- DERMOSCOPY ++++\p
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2

mypath=os.getcwd()

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
cnt=0
border=5
plt.close('all')
for i in onlyfiles:
    I=False
    if 'png' in i  or  'jpg' in i or  'PNG' in i or  'JPG' in i :
        I=True
    if not '.py' in i and not 'croped_' in i and  I :
        try:
            cnt=cnt+1
            print(i)
            a0 = plt.imread(i)
            a0_cv2 = cv2.imread(i)
            
            plt.figure(cnt)
            plt.subplot(221)
            plt.imshow(a0,'gray')
            
            
            a= (a0[:,:,0]+a0[:,:,1]+a0[:,:,2])
            a00=a[0,0]
            
            a1=np.sum(a,1)
            a2=np.sum(a,0)
            plt.subplot(222)
            plt.plot(a1)
            plt.subplot(223)
            plt.plot(a2)
            cnt1_1=0
            cnt1_2=1
            cnt2_1=0
            cnt2_2=1
            
            while a1[cnt1_1] == a1[0]:
                cnt1_1=cnt1_1+1
            while a2[cnt2_1] == a2[0]:
                cnt2_1=cnt2_1+1
                
            
            while a1[-cnt1_2] == a1[0]:
                cnt1_2=cnt1_2+1
            while a2[-cnt2_2] == a2[0]:
                cnt2_2=cnt2_2+1    
            print (cnt1_1,  cnt1_2,    cnt2_1, cnt2_2)
            plt.subplot(224);
            
            
            if cnt1_1<border:cnt1_1=0
            if cnt1_1>= border:cnt1_1=cnt1_1 -border
            
            if cnt2_1<border:cnt2_1=0
            if cnt2_1>= border:cnt2_1=cnt2_1 -border
            
            
            if cnt1_2<border:cnt1_2=1
            if cnt1_2>= border:cnt1_2=cnt1_2 -border
            
            
            if cnt2_2<border:cnt2_2=1
            if cnt2_2>= border:cnt2_2=cnt2_2 -border
            
            a3=a0_cv2[cnt1_1: -cnt1_2,    cnt2_1:-cnt2_2,: ]
            
            plt.imshow(a3, 'gray')
    
            # a3=a[cnt1_1: -cnt1_2,    cnt2_1:-cnt2_2 ]
            
            filename='croped_'+i
            cv2.imwrite(filename, a3)
        except: print ('Error')