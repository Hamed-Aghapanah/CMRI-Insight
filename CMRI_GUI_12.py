
show_authors=False
show_authors=True


time_phase1 = 3000 ### mili seconds
time_phase2 = 20 ### mili seconds
step_phase2 = 5 ### mili seconds
step_phase3 = 20 ### mili seconds
time_phase3 = 60 ### mili seconds



import datetime
from time import gmtime, strftime
import sys
import cv2
from PyQt5 import QtWidgets, QtGui
# from PIL import Image, ImageQt
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QApplication, QColorDialog,QFontDialog
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore
# create window here...
# window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

# =============================================================================
# warrning off
# =============================================================================
import warnings
import sys
# import shutup; shutup.please()
if not sys.warnoptions:
    warnings.simplefilter("ignore")
def fxn():warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings("ignore")
warnings.warn('my warning')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")
plt.close('all')
import warnings
warnings.filterwarnings('ignore')


import warnings
warnings.filterwarnings('default')
import pandas as pd
pd.options.mode.chained_assignment = None
import pandas as pd
pd.options.mode.chained_assignment = 'warn'
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# =============================================================================
# def
# =============================================================================
def dice_function2 (Y,YY1):           
    d_r=1; d_l=1;      d_m=1;d_b=1;
    B1= Y[:,:,1]    + Y[:,:,0]   +Y[:,:,2]
    B2= YY1[:,:,1]  + YY1[:,:,0] +YY1[:,:,2]
    i,j = np.where(B1>1) ; B1 [i,j]=1;B1=1-B1
    i,j = np.where(B2>1) ; B2 [i,j]=1;B2=1-B2
    epsilon=0.001
    for i in range(np.shape(B1)[0]):
        for j in range(np.shape(B1)[1]):
            if B1[i,j]>1:B1[i,j]=1
            if B2[i,j]>1:B2[i,j]=1
    if sumer ( Y[:,:,0])>0 :
        d_r = ( 100*2* sumer(dot_multi( Y[:,:,0],YY1[:,:,0])) /sumer(( Y[:,:,0]+YY1[:,:,0])))/100  # d_r =int ( 100*1* sumer(dot_multi( Y[:,:,0],YY1[:,:,0])) /sumer(community( Y[:,:,0],YY1[:,:,0])))/100
    if sumer ( Y[:,:,1])>0 :
        d_l = ( 100*2* sumer(dot_multi( Y[:,:,1],YY1[:,:,1])) /sumer(( Y[:,:,1]+YY1[:,:,1])))/100 # d_l =int ( 100*1* sumer(dot_multi( Y[:,:,1],YY1[:,:,1])) /sumer(community( Y[:,:,1],YY1[:,:,1])))/100
    if sumer ( Y[:,:,2])>0 :
        d_m = ( 100*2* sumer(dot_multi( Y[:,:,2],YY1[:,:,2])) /sumer(( Y[:,:,2]+YY1[:,:,2])))/100 # d_m =int ( 100*1* sumer(dot_multi( Y[:,:,2],YY1[:,:,2])) /sumer(community( Y[:,:,2],YY1[:,:,2])))/100
    try:
        d_b = ( 100*2* sumer(dot_multi( 1-B1,1-B2)) /sumer((1-B1+1-B2)))/100
    except:s=1  
    d_r =np.round(100*(d_r+epsilon))/100
    d_l =np.round(100*(d_l+epsilon))/100
    d_m =np.round(100*(d_m+epsilon))/100
    d_b = ( 100*2* sumer(dot_multi( B1,B2)) /sumer((B1+B2)))/100
    
    d_d_bm =np.round(100*(d_b+epsilon))/100
    d_overal= ( (d_r+d_l+d_m+d_b))/4
    d_b =np.ceil(100*d_b)/100
    
    dice_RV=d_r
    dice_LV=d_l
    dice_Myo=d_m
    mse1=np.mean(np.abs(Y-YY1))
    
    return dice_RV,dice_LV,dice_Myo,mse1,d_b 
# =============================================================================
# convert ui to py
# =============================================================================


# from G2 import *
# from G3 import *
# from G4 import *
# from G5 import *
# from G6 import *
# from G7 import *
# from G8 import *
# from G9 import *
# from G10 import *
from GUI_dicom12 import *

import cv2
import numpy as np


if show_authors :
    from subprocess import call
    call(["python", "convert_ui_to_py.py"])
    src1=cv2.imread( 'images/Slide1.PNG')
    image0 = cv2.resize(src1, (840, 680))
    
    TEXT=115*' '+' CIAG Group'
    cv2.imshow(TEXT,image0)
    cv2.waitKey(time_phase1)
    
    cv2.destroyWindow(TEXT)



class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.time000=time_phase2
        self.datee=0
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        shamsi='1402 03-05'
        
        
        self.mask_e=0
        self.image_e=0
        self.predicted_e=0
        
        self.mask=0
        self.image=0
        self.predicted=0
        
        self.model1_e=0
        self.model2_e=0
        self.model3_e=0
        self.model4_e=0
        self.model5_e=0
        self.model6_e=0


        self.mask=0
        self.out=0
        # self.ui.segmentor1.setEnabled(True)
        # self.ui.segmentor2.setEnabled(True)
        # self.ui.segmentor3.setEnabled(True)
        # self.ui.segmentor5.setEnabled(True)
        # self.ui.segmentor6.setEnabled(True)
        
        self.dice_all=0
        self.dice_RV=0
        self.dice_LV=0
        self.dice_Myo=0
        self.mse=0

        self.ui.result_seg_box.setEnabled(False);
        # self.ui.dice_overal.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.cal_res1.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.dice_RV.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.dice_Myo.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.mse1.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        
        # self.ui.segmentor6.setEnabled(True)


        
        # self.ui.class_1.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        # self.ui.class_2.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        # self.ui.class_3.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        # self.ui.class_4.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        # self.ui.class_5.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        

        self.ui.progressBar_segmentation.setValue(0)
        self.ui.progressBar_classification.setValue(0)
        self.ui.progressBar_mesh.setValue(0)
        # self.ui.result_box.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(123, 176, 255); font: 75 12pt }")
        
        
        # self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.result_seg_box.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        
        
        self.ui.actionHelp_Software.triggered.connect(self.Help_Software_function)

        self.ui.actionVisit_software_page.triggered.connect(self.Visit_Our_site_function)
        self.ui.actionLicence.triggered.connect(self.Licence_function)
        
        
        self.ui.fig1.setEnabled(False);self.ui.fig2.setEnabled(False)
        self.ui.fig3.setEnabled(False);self.ui.fig4.setEnabled(False)
        self.setWindowTitle(' Cardiac MR Images Analysis Toolkit  (CIAG)')
        self.show()
# =============================================================================
# section 0 : banner and initial images
# =============================================================================
        self.counter1=0
        self.counter2=0
        self.stage = 1
        self.th_pre=0
        self.pre = 0
        self.image1=0
        self.image2=0
        # self.seg=1
        self.th=1  
        self.Author=0
        self.deep=0
        self.zoom1=1
        self.zoom2=1
        col = QColor(74,207,53)
        self.ui.time1.setEnabled(True)
        self.ui.time2.setEnabled(True)
        # self.ui.time_p.setEnabled(True)
        import cv2
        image0=cv2.imread('images/banner 2.jpg')
        # self.dis1(image0)
        flag=cv2.imread('flag/Britain-512.png')
        self.dis_flag(flag)
        
        # logo=cv2.imread('images/logo.png')
        # self.dis_logo(logo)
        
        image000=cv2.imread('images/banner 3.jpg')
        image_arm=cv2.imread('images/arm.jpg')
        image_author=cv2.imread('images/author.jpg')

        self.dis1(image0)
        self.dis2(image0)
        self.dis_mesh(image0)
        self.dis_arm(image_arm)
        
        self.dis_result(image0)
        self.dis_gg(image0)
        self.dis_author(image_author)
        self.classifer='Deep learning'
        import time
        from time import gmtime, strftime
        import calendar
        import time
        import numpy as np
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
                       
        print('T1 = ',T1)
        print('T2 = ',T2)
        Y=T2[0:4];m=T2[5:7];d=T2[8:10];
        
        Y=int(Y)
        m=int(m)
        d=int(d)
         
        a=calendar.month(Y, m )
        self.ui.time1.setText(T1)
        self.ui.time2.setText(T2)
        try:
            print ('shamsi 0' + shamsi)
            from time import gmtime, strftime
            T2=strftime("%Y-%m-%d ", time.localtime());Y=T2[0:4];m=T2[5:7];d=T2[8:10];
            Y=int(Y);         m=int(m);         d=int(d)
            print ('shamsi 0 T2' + T2)
            import jdatetime             # gregorian_date = jdatetime.date(1396,2,30).togregorian()
            shamsi =  str(jdatetime.date.fromgregorian(day=d,month=m,year=y) )
            print ('shamsi00' + shamsi)
        except:s=1
        
        print ('shamsi2' + shamsi)
        self.ui.time_p.setText(shamsi)
        try:
            import jdatetime
            T3 = jdatetime.date.fromgregorian(day = d, month = m, year = Y )  
            self.ui.time_p.setText(T3)
        except:s=1
        # self.ui.dateee.setText(a)
            
        self.ui.time1.setText(T1)
        self.ui.time2.setText(T2)
        self.scene = QGraphicsScene(self)

        self.ui.Author.clicked.connect(self.Author_function)
        self.ui.pushButtonColor1.clicked.connect(self.dispcolor1)
        self.ui.pushButtonColor2.clicked.connect(self.dispcolor2)
        self.ui.pushButtonFont.clicked.connect(self.pushButtonfont_function)
        self.ui.pushButtonLanguage.clicked.connect(self.pushButtonLanguage_function)
        self.ui.pushButtonTheme.clicked.connect(self.pushButtonTheme_function)

        self.ui.fullscreen1.clicked.connect(self.fullscreen1_function)
        self.ui.fullscreen2.clicked.connect(self.fullscreen2_function)     
        self.ui.saveimage.clicked.connect(self.saveimage_function)
        self.ui.saveimage_2.clicked.connect(self.saveimage_2_function)
        self.ui.time1.clicked.connect(self.time1_function)
        # self.ui.time2.clicked.connect(self.time2_function)
# =============================================================================
# section 1 : showing 
# =============================================================================
        self.ui.loadimage.clicked.connect(self.loadimage_function)
        self.ui.load_mask.clicked.connect(self.load_mask_function)
        self.ui.DICOM_input.clicked.connect(self.DICOM_input_function)
        self.ui.cropeimage.clicked.connect(self.cropeimage_function)
        self.ui.cropeimage_2.clicked.connect(self.cropeimage_2_function)

# =============================================================================
# section 2 : preprocessing         
# =============================================================================
       
# =============================================================================
# section 3 : Segmentation         
# =============================================================================
        SEGMENTOR_LIST=['CardSegNet','CardUnet ',
                        'MSTGANet','CE-Net',
                        'UNet++','nnUnet ',
                        'UNet','CE-Deeplab v3',
                        ]
        self.ui.segmentor1.toggled.connect(self.dispAmount3)
        self.ui.segmentor2.toggled.connect(self.dispAmount3)
        self.ui.segmentor3.toggled.connect(self.dispAmount3)
        self.ui.segmentor4.toggled.connect(self.dispAmount3)
        self.ui.segmentor5.toggled.connect(self.dispAmount3)
        self.ui.segmentor6.toggled.connect(self.dispAmount3)
        self.ui.segmentor7.toggled.connect(self.dispAmount3)
        self.ui.segmentor8.toggled.connect(self.dispAmount3)
        
        # self.ui.manualsegmentation.clicked.connect(self.manualsegmentation_function)
        self.ui.applysegmentation.clicked.connect(self.applysegmentation_function)
        self.ui.load_models.clicked.connect(self.load_models_function)
        # self.ui.segmentationbar.valueChanged.connect(self.segmentationbar_function)
# =============================================================================
# section 4 : Input data         
# =============================================================================
        self.ui.cal_res1.clicked.connect(self.cal_res1_function)
    
# =============================================================================
# section 5 : Clinical ABCD        
# =============================================================================
    
# =============================================================================
# section 6 : Classification         
# =============================================================================
        self.ui.applyclassification.clicked.connect(self.applyclassification_function)
        self.ui.classifier1.toggled.connect(self.dispAmount2)
        self.ui.classifier2.toggled.connect(self.dispAmount2)
        self.ui.classifier3.toggled.connect(self.dispAmount2)
        self.show()
# =============================================================================
# section 7 : results         
# =============================================================================
    
# =============================================================================
# section 8 : others         
# =============================================================================
#        self.ui.pushButtonCreateDB.clicked.connect(self.createDatabase)
# =============================================================================
# =============================================================================
    @pyqtSlot()
# =============================================================================
#     f0
# =============================================================================
       
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        f = open(fileName,'w')
        text = self.ui.textEdit.toPlainText()
        f.write(text)
        f.close()
        
    def about_us_function(self):
        import webbrowser
        webbrowser.open('https://www.hamedaghapanah.com/index.php?action_skin_change=yes&skin_name=en')
    def Help_Software_function(self):
        import webbrowser
        webbrowser.open('https://www.hamedaghapanah.com/index.php?newsid=18')
    def Visit_Our_site_function(self):
        import webbrowser
        webbrowser.open('https://www.hamedaghapanah.com/')
    def Licence_function(self):
        import webbrowser
        webbrowser.open('https://www.hamedaghapanah.com/index.php?newsid=3')
     
    def dispcolor1(self):
        col = QColorDialog.getColor()
        if col.isValid():
            
            self.ui.fig0.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.seg_text1.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.seg_text2.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.seg_text3.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.seg_text4.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.zoom_bar_1.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.zoom_bar_2.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.zoom_bar_1.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.zoom_bar_2.setStyleSheet("QWidget { background-color: %s }" % col.name())
            
             
            self.ui.label_zoom1.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.label_zoom2.setStyleSheet("QWidget { background-color: %s }" % col.name())
    def dispcolor2(self):
        col = QColorDialog.getColor()
#        print(col)
        if col.isValid():          
            self.ui.themebox.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            ## self.ui.clinical_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            #self.ui.input_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.classification_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.result_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.result_seg_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            # self.ui.calendarWidget.setStyleSheet("QWidget { background-color: %s }" % col.name())


    def pushButtonfont_function(self):
        font, ok = QFontDialog.getFont()
        if ok:
            self.ui.loadimage.setFont(font)  
            self.ui.status.setText(' font is changed ')
        
    def pushButtonLanguage_function(self):
        lan_index=self.ui.languageComboBox.currentIndex()
        lan=['English','فارسی','French','Germany','Hindustani','Chinese','Spanish','العربیه','Malay','Russian','italian','Dutch']
        lan0=lan[lan_index]
        print(' language is changed to '+lan0)
        self.ui.status.setText(' language is changed to '+lan0 )
        
        
        
        if lan0=='English':
            
            flag=cv2.imread('flag/Britain-512.png')
            self.dis_flag(flag)
            languager=['Load Image','DICOM','load mask',' Images + Masks','Zoom','Crop' ,
                  'Apply', 'load models',
                  'Deep learning','classifier2 on Deep Features','classifier3 on Deep Features',
                  'Apply','Original Image','Result Image','Save Image','Full Screen','Save Image',
                  'Full Screen',
                  'Created by Hamed Aghapanah    Isfahan University of Medical Science     Version 3.1.1 , 2024',
                  'Background','Color','Boxes','Font' ,
                  'Dilated CardioMyopathy (DCM)',
                  'Hypertrophic CardioMyopathy (HCM)',
                  'Myocardial Infarction (MI)',
                  'Anomalous Right Ventricle (ARV)',
                  'Normal Controls (Healthy)'    
                  ]
            
            
            self.language_function(languager)
            
             
        if lan0=='French':
            flag=cv2.imread('flag/french-512.png')
            self.dis_flag(flag)
            
            
            languager=['Charger l image','DICOM','charger le masque',' Images + Masques','Zoom','Rogner' ,
                               'Appliquer', 'charger les modèles',
                               'Apprentissage en profondeur','classificateur2 sur les fonctionnalités profondes','classificateur3 sur les fonctionnalités profondes',
                               'Appliquer','Image d origine','Image de résultat','Enregistrer l image','Plein écran','Enregistrer l image',
                               'Plein écran',
                               'Créé par Hamed Aghapanah Isfahan University of Medical Science Version 3.1.1, 2024',
                               'Arrière-plan','Couleur','Boîtes','Police' ,
                               'CardioMyopathie dilatée (DCM)',
                               'CardioMyopathie Hypertrophique (HCM)',
                               'Infarctus du myocarde (IM)',
                               'Ventricule droit anormal (ARV)',
                               "Contrôles normaux (sains)"
                               ]
            
            
            self.language_function(languager)
                        
            
            
        if lan0=='Germany':
            flag=cv2.imread('flag/German_512.png')
            self.dis_flag(flag) 
            
            
            
            
            languager=['Bild laden', 'DICOM', 'Maske laden', 'Bilder + Masken', 'Zoom', 'Zuschneiden' ,
                               'Anwenden', 'Modelle laden',
                               'Deep Learning', 'classifier2 für Deep Features', 'classifier3 für Deep Features',
                               'Anwenden', 'Originalbild', 'Ergebnisbild', 'Bild speichern', 'Vollbild', 'Bild speichern',
                               'Ganzer Bildschirm',
                               'Erstellt von Hamed Aghapanah Isfahan University of Medical Science Version 3.1.1, 2024',
                               'Hintergrund','Farbe','Boxen','Schriftart',
                               'Dilatative KardioMyopathie (DCM)',
                               'Hypertrophe KardioMyopathie (HCM)',
                               'Myokardinfarkt (MI)',
                               'Anomaler rechter Ventrikel (ARV)',
                               'Normale Kontrollen (gesund)'
                               ]
            
            
            self.language_function(languager)

             
        if lan0=='Hindustani':
            flag=cv2.imread('flag/India-512.png')
            self.dis_flag(flag)  
            
            
            
            
            languager=['लोड इमेज', 'डीकॉम', 'लोड मास्क', 'इमेज + मास्क', 'ज़ूम', 'क्रॉप',
                        'लागू करें', 'लोड मॉडल',
                   'डीप लर्निंग', 'क्लासिफायर2 ऑन डीप फीचर्स', 'क्लासिफायर3 ऑन डीप फीचर्स',
                   'लागू करें', 'मूल छवि', 'परिणाम छवि', 'छवि सहेजें', 'पूर्ण स्क्रीन', 'छवि सहेजें',
                   'पूर्ण स्क्रीन',
                   'हमीद अघपनाह इस्फ़हान यूनिवर्सिटी ऑफ़ मेडिकल साइंस संस्करण 3.1.1, 2024 द्वारा बनाया गया',
                   'पृष्ठभूमि', 'रंग', 'बक्से', 'फ़ॉन्ट',
                   'दिलित कार्डियोमायोपैथी (डीसीएम)',
                   'हाइपरट्रॉफिक कार्डियोमायोपैथी (एचसीएम)',
                   'मायोकार्डिअल इन्फ्रक्शन (एमआई)',
                   'विषम दायां वेंट्रिकल (एआरवी)',
                   'सामान्य नियंत्रण (स्वस्थ)'
                   ]
            
            
            self.language_function(languager)

             
        if lan0=='Chinese':
            flag=cv2.imread('flag/China-512.png')
            self.dis_flag(flag) 
            
            
            
            
            languager=['加载图片','DICOM','加载掩码','图片+面具','缩放','裁剪',
                               '应用','加载模型',
                               '深度学习','深度特征分类器2','深度特征分类器3',
                               '应用','原始图像','结果图像','保存图像','全屏','保存图像',
                               '全屏',
                               '由 Hamed Aghapanah Isfahan 医科大学创建,第 3.1.1 版,2024 年',
                               '背景','颜色','方框','字体',
                               '扩张型心肌病 (DCM)',
                               '肥厚性心肌病 (HCM)',
                               '心肌梗塞 (MI)',
                               '异常右心室 (ARV)',
                               '正常对照（健康）'
                               ]
            
            
            self.language_function(languager)
 
             
        if lan0=='Spanish':
            flag=cv2.imread('flag/Spain-2-512.png')
            self.dis_flag(flag) 
            
            
            
            
            languager=['Cargar imagen','DICOM','cargar máscara','Imágenes + Máscaras','Zoom','Recortar' ,
                        'Aplicar', 'cargar modelos',
                   'Aprendizaje profundo', 'clasificador2 en funciones profundas', 'clasificador3 en funciones profundas',
                   'Aplicar', 'Imagen original', 'Imagen de resultado', 'Guardar imagen', 'Pantalla completa', 'Guardar imagen',
                   'Pantalla completa',
                   'Creado por Hamed Aghapanah Isfahan University of Medical Science Versión 3.1.1, 2024',
                   'Fondo','Color','Cuadros','Fuente' ,
                   'Miocardiopatía dilatada (MCD)',
                   'Miocardiopatía Hipertrófica (MCH)',
                   'Infarto de miocardio (IM)',
                   'Ventrículo derecho anómalo (ARV)',
                   'Controles normales (saludables)'
                   ]
            
            
            self.language_function(languager)
 
             
        if lan0=='العربیه':
            flag=cv2.imread('flag/Saudia_arabia_national_flags_country_flags-512.png')
            self.dis_flag(flag) 
            
            
            
            languager=['تحميل الصورة' , 'DICOM' , 'تحميل القناع' , 'الصور + الأقنعة' , 'تكبير' , 'اقتصاص' ,
                                    'تطبيق' , 'تحميل النماذج' ,
                               'التعلم العميق' , 'المصنف 2 على الميزات العميقة' , 'المصنف 3 على الميزات العميقة' ,
                               'تطبيق' , 'الصورة الأصلية' , 'الصورة الناتجة' , 'حفظ الصورة' , 'ملء الشاشة' , 'حفظ الصورة' ,
                               'تكبير الشاشة',
                               'من إعداد حامد أغابانا , جامعة أصفهان للعلوم الطبية , الإصدار الثاني , 2024' ,
                               'الخلفية' , 'اللون' , 'المربعات' , 'الخط' ,
                               'تمدد عضلة القلب (DCM)' ,
                               'اعتلال عضلي القلب الضخامي (HCM)' ,
                               'احتشاء عضلة القلب (MI)',
                               'البطين الأيمن الشاذ (ARV)',
                               'الضوابط العادية (صحية)'
                               ]
            
            
            self.language_function(languager)

             
        if lan0=='Malay':
            flag=cv2.imread('flag/flag-39-512.png')
            self.dis_flag(flag) 
            
            
            
            languager=['Muat Imej','DICOM','muat topeng',' Imej + Topeng','Zum','Pangkas' ,
                        'Gunakan', 'muat model',
                   'Pembelajaran mendalam','pengkelas2 pada Ciri Mendalam','pengkelas3 pada Ciri Mendalam',
                   'Gunakan','Imej Asal','Imej Hasil','Simpan Imej','Skrin Penuh','Simpan Imej',
                   'Skrin penuh',
                   'Dicipta oleh Universiti Sains Perubatan Hamed Aghapanah Isfahan Versi 3.1.1 , 2024',
                   'Latar Belakang', 'Warna', 'Kotak', 'Font' ,
                   'Dilated CardioMyopathy (DCM)',
                   'Hypertrophic CardioMyopathy (HCM)',
                   'Infarksi Miokardium (MI)',
                   ' Ventrikel Kanan Anomali (ARV)',
                   'Kawalan Normal (Sihat)'
                   ]
            
            
            self.language_function(languager)

             
        if lan0=='Russian':
            flag=cv2.imread('flag/Russian-512.png')
            self.dis_flag(flag)
             
            
            
            languager=['Загрузить изображение','DICOM','загрузить маску','Изображения + маски','Масштаб','Обрезать',
                                    'Применить', 'загрузить модели',
                               'Глубокое обучение', 'классификатор2 по глубоким функциям', 'классификатор3 по глубоким функциям',
                               'Применить', 'Исходное изображение', 'Изображение результата', 'Сохранить изображение', 'Во весь экран', 'Сохранить изображение',
                               'Полноэкранный',
                               'Создано Хамедом Агапанахом Исфаханским университетом медицинских наук, версия 3.1.1, 2024 г.',
                               'Фон', 'Цвет', 'Прямоугольники', 'Шрифт',
                               'Дилатационная кардиомиопатия (ДКМ)',
                               'Гипертрофическая кардиомиопатия (ГКМП)',
                               'Инфаркт миокарда (ИМ)',
                               'Аномальный правый желудочек (АРВ)',
                               'Нормальный контроль (здоровый)'
                               ]
            
            
            self.language_function(languager)

        if lan0=='italian':
            flag=cv2.imread('flag/Italy-512.png')
            self.dis_flag(flag)
            
             
            
            
            languager=['Carica immagine','DICOM','carica maschera',' Immagini + Maschere','Zoom','Ritaglia' ,
                        'Applica', 'carica modelli',
                   'Deep learning','classifier2 su Deep Features','classifier3 su Deep Features',
                   'Applica','Immagine originale','Immagine risultato','Salva immagine','Schermo intero','Salva immagine',
                   'A schermo intero',
                   'Creato da Hamed Aghapanah Isfahan University of Medical Science Version 3.1.1 , 2024',
                   'Sfondo','Colore','Riquadri','Carattere' ,
                   "Cardiomiopatia dilatativa (DCM)",
                   "Cardiomiopatia ipertrofica (HCM)",
                   'Infarto del miocardio (IM)',
                   'Ventricolo destro anomalo (ARV)',
                   'Controlli normali (sani)'
                   ]
            
            
            self.language_function(languager)

        if lan0=='فارسی':
            flag=cv2.imread('flag/iran-512.png')
            self.dis_flag(flag)
            languager=[  'بارگیری تصویر', 'دایکام', 'بارگذاری ماسک', 'تصاویر + ماسک‌ها', 'زوم', 'برش',
                                'اعمال', 'مدل های بارگیری',
                           'آموزش عمیق', 'classifier2', 'classifier3 ',
                           'اعمال', 'تصویر اصلی', 'تصویر نتیجه', 'ذخیره تصویر', 'تمام صفحه', 'ذخیره تصویر',
                           'تمام صفحه',
                           'تهیه شده توسط حامد آقا پناه دانشگاه علوم پزشکی  اصفهان نسخه 3.1.1  2024',
                           'پس زمینه', 'رنگ', 'جعبه ها', 'قلم',
                           'کاردیومیوپاتی گشاد شده (DCM)',
                           'کاردیومیوپاتی هیپرتروفیک (HCM)',
                           'انفارکتوس میوکارد (MI)',
                           'بطن راست غیرعادی (ARV)',
                           'کنترل های عادی (سالم)'
                  ]
            
            
            self.language_function(languager)
            
        if lan0=='Dutch':
            flag=cv2.imread('flag/Dutch_512.png')
            self.dis_flag(flag)      
             
            
            
            languager=['Afbeelding laden','DICOM','masker laden',' Afbeeldingen + maskers','Zoom','Bijsnijden',
                        'Toepassen', 'modellen laden',
                   'Deep learning','classifier2 op Deep Features','classifier3 op Deep Features',
                   'Toepassen','Oorspronkelijke afbeelding','Resultaatafbeelding','Afbeelding opslaan','Volledig scherm','Afbeelding opslaan',
                   'Volledig scherm',
                   'Gemaakt door Hamed Aghapanah Isfahan University of Medical Science Versie 2, 2024',
                   'Achtergrond','Kleur','Vakjes','Lettertype',
                   'Uitgezette cardiomyopathie (DCM)',
                   'Hypertrofische cardiomyopathie (HCM)',
                   'Myocardinfarct (MI)',
                   'Abnormaal Rechterventrikel (ARV)',
                   'Normale controles (gezond)'
                   ]
            
            
            self.language_function(languager)


    def pushButtonTheme_function(self):
        theme_index=self.ui.ThemeComboBox.currentIndex()
        col1=0x0000008C725B2F98
        col2=0x0000008C725B2F98   
## bold
#Black 	30 	No effect 	0 	Black 	40
#Red 	31 	Bold 	1 	Red 	41
#Green 	32 	Underline 	2 	Green 	42
#Yellow 	33 	Negative1 	3 	Yellow 	43
#Blue 	34 	Negative2 	5 	Blue 	44
#Purple 	35 			Purple 	45
#Cyan 	36 			Cyan 	46
#White 	37 			White 

        if theme_index==0:
            blocks=['fig0','seg_text1','seg_text2','seg_text3','seg_text4','seg_text5','seg_text6','seg_text7','seg_text8',
                    'zoom_bar_1','zoom_bar_2','zoom_bar_3','zoom_bar_4',
                    'label_zoom1','label_zoom2','label_zoom3','label_zoom4',
                    'time2','time1','time_p','result_seg_box','classification_box','result_box']
                    # 'time2','time1','time_p','result_seg_box','classification_box','result_box','calendarWidget']
            
            self.color_b_function(blocks, 'black')
            blocks=['loadimage','DICOM_input','cropeimage','cropeimage_2',
                    'pushButtonColor1','pushButtonColor2','load_models',
                    'applysegmentation','load_models','saveimage','pushButtonTheme',
                    'pushButtonColor1','pushButtonColor2',
                    'pushButtonFont','pushButtonLanguage','languageComboBox']
                
            self.color_b_function(blocks, 'Green')
                         
                        
            

            self.ui.themebox.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color:rgb(87, 112, 255) }" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" )             
            
            self.ui.result_seg_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" ) 
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" ) 
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" ) 
            # self.ui.calendarWidget.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" ) 
            
            
        if theme_index==1:
            self.ui.fig0.setStyleSheet("QWidget { background-color:white}")
            blocks=['fig0','seg_text1','seg_text2','seg_text3','seg_text4','seg_text5','seg_text6','seg_text7','seg_text8',
                    'zoom_bar_1','zoom_bar_2','zoom_bar_3','zoom_bar_4',
                    'label_zoom1','label_zoom2','label_zoom3','label_zoom4',
                    # 'time2','time1','time_p','result_seg_box','classification_box','result_box','calendarWidget']
                    'time2','time1','time_p','result_seg_box','classification_box','result_box']
            
            self.color_b_function(blocks, 'white')
            blocks=['loadimage','DICOM_input','cropeimage','cropeimage_2',
                    'pushButtonColor1','pushButtonColor2','load_models',
                    'applysegmentation','load_models','saveimage','pushButtonTheme',
                    'pushButtonColor1','pushButtonColor2',
                    'pushButtonFont','pushButtonLanguage','languageComboBox']
                
           
            
            self.ui.themebox.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: bold gray}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color: rgb(187, 212, 255)}" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(187, 212, 255)}" )             

            self.ui.loadimage.setStyleSheet("QWidget { background-color: Cyan }" )             
            self.ui.DICOM_input.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.cropeimage.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.cropeimage_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.saveimage.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Cyan}" )             
           
            self.ui.applysegmentation.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.load_models.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.applyclassification.setStyleSheet("QWidget { background-color: Cyan}" )             
            
            self.ui.pushButtonTheme.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonFont.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonLanguage.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.languageComboBox.setStyleSheet("QWidget { background-color: Cyan}" ) 
 
            
        if theme_index>1:      
            
#            pushButtonColor1
            self.ui.fig0.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            
            self.ui.seg_text1.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.seg_text2.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.seg_text3.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.seg_text4.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.seg_text5.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.seg_text6.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            # self.ui.seg_text7.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.seg_text8.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            
            
            self.ui.zoom_bar_1.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.zoom_bar_2.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.zoom_bar_3.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.zoom_bar_4.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.label_zoom1.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.label_zoom2.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.label_zoom3.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
            self.ui.label_zoom4.setStyleSheet("QWidget {background-color: rgb(150, 150, 150);}")
    
            self.ui.themebox.setStyleSheet("QWidget { background-color: rgb(194, 133, 255)}" )
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.495, cy:0.494318, radius:2, fx:0.489, fy:0.494318, stop:0 rgba(221, 32, 56, 255), stop:1 rgba(255, 255, 255, 255));}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: rgb(231, 144, 255);}" )
            # self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(194, 133, 255);}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(121, 123, 255)}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
            self.ui.time_p.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
            # self.ui.author_7.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
 
            self.ui.loadimage.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255)); }" )             
            self.ui.DICOM_input.setStyleSheet("QWidget {  background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255));}")             
            self.ui.cropeimage.setStyleSheet("QWidget {  background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255))};" )             
            self.ui.cropeimage_2.setStyleSheet("QWidget {  background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255))};" )             
            self.ui.saveimage.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255));}" )             
            self.ui.pushButtonFont.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" )             
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 

            self.ui.applysegmentation.setStyleSheet("QWidget { background-color: rgb(231, 194, 255);}" )             
            self.ui.load_models.setStyleSheet("QWidget { background-color: rgb(231, 194, 255);}" )             
            self.ui.applyclassification.setStyleSheet("QWidget {background-color: rgb(171, 173, 255);}" )             
            #            pushButtonFont
            self.ui.pushButtonTheme.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonLanguage.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.languageComboBox.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            
            self.ui.result_seg_box.setStyleSheet("QWidget { background-color: rgb(170, 130, 255);}" ) 
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(121, 123, 255);}" ) 
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" ) 
            # self.ui.calendarWidget.setStyleSheet("QWidget { background-color: rgb(121, 255, 150);}" ) 
           
        theme=['Dark','Bright','Default']
        theme0=theme[theme_index]
        self.ui.status.setText(' theme is changed to '+theme0 )
# =============================================================================
#   timer 
# ============================================================================= 

    def time1_function(self):
        from subprocess import call
        call(["python", "clockk.py"])
        import time
        from time import gmtime, strftime
        import calendar
        import time
        import numpy as np
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
                       
        print('T1 = ',T1)
        print('T2 = ',T2)
        Y=T2[0:4];m=T2[5:7];d=T2[8:10];
        
        Y=int(Y)
        m=int(m)
        d=int(d)
        
        print(T1)
        self.ui.time1.setText(T1)
        shamsi='1402 03-05'
        try:
            import jdatetime             # gregorian_date = jdatetime.date(1396,2,30).togregorian()
            shamsi =  str(jdatetime.date.fromgregorian(day=d,month=m,year=y) )
            print ('shamsi' + shamsi)
        except:s=1
        print ('shamsi3' + shamsi)
        self.ui.time_p.setText(shamsi)
        try:
            import jdatetime
            T3 = jdatetime.date.fromgregorian(day = d, month = m, year = Y )  
            self.ui.time_p.setText(T3)
        except:s=1
        
        
        
        
    def time2_function(self):
                
        import time
        from time import gmtime, strftime
        import calendar
        import time
        import numpy as np
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
                       
        print('T1 = ',T1)
        print('T2 = ',T2)
        Y=T2[0:4];m=T2[5:7];d=T2[8:10];
        
        Y=int(Y)
        m=int(m)
        d=int(d)
        
        a=calendar.month(Y, m)
        print(a)
        self.ui.time1.setText(T1)
        self.ui.time2.setText(T2)
        try:
            import jdatetime             # gregorian_date = jdatetime.date(1396,2,30).togregorian()
            shamsi =  str(jdatetime.date.fromgregorian(day=d,month=m,year=y) )
            print ('shamsi' + shamsi)
        except:s=1
        print ('shamsi8' + shamsi)
        self.ui.time_p.setText(shamsi)
        try:
            import jdatetime
            T3 = jdatetime.date.fromgregorian(day = d, month = m, year = Y )  
            self.ui.time_p.setText(T3)
        except:s=1
        
        
        if self.datee==1:
            self.datee=0
            # self.ui.dateee.setText(a)
        else:
            self.datee=1
            # self.ui.dateee.setText('Calendar')

# =============================================================================
#   f1
# ============================================================================= 
    def loadimage_function(self):
        self.ui.progressBar_segmentation.setValue(0)
        # self.mask=0
        blocks=['loadimage','DICOM_input','cropeimage_2','cropeimage','applysegmentation','applysegmentation','load_models',
        'saveimage','fullscreen1','saveimage_2','fullscreen2','pushButtonTheme','pushButtonColor2','pushButtonLanguage','pushButtonFont','time1','time2','time_p',
        'pushButtonColor1','pushButtonColor2','pushButtonLanguage','pushButtonFont',
        'languageComboBox' 'ThemeComboBox',]
        self.enable_b_function(blocks,'False')
        
         
        import os
        path0= os.getcwd()
        fname = QFileDialog.getOpenFileName(self, 'Open file', path0)
        print(fname)
        if fname[0] !='':
            import cv2
            import numpy as np
            image0=cv2.imread(fname[0])          
            self.image0 = image0
            self.dis1(image0)
            self.stage=3
            stage =self.stage
            self.stage_function(stage)
            self.dis1(image0)
            self.image= image0
            import cv2
            font = cv2.FONT_HERSHEY_COMPLEX
            import copy
            p1=int(np.fix(np.shape(image0) [0] /2  ))
            
            for font1 in range (step_phase2):
                for rep in range(2):
                    f1=0.01 * np.shape(image0) [0] * font1/step_phase2
                    f2=0.008 * np.shape(image0) [0] * font1/step_phase2
                    image0_add_mask=copy.deepcopy(image0)
                    image0_add_mask =cv2.putText(image0_add_mask,' image',(0,p1),font,f1,(255,0,255),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis1(image0_add_mask)
                    
                    
                    image0_add_mask=copy.deepcopy(image0)
                    image0_add_mask =cv2.putText(image0_add_mask,' MASK',(0,p1),font,f1,(0,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis2(image0_add_mask)
                    image0_add_grad_cam=copy.deepcopy(image0)
                    image0_add_grad_cam =cv2.putText(image0_add_grad_cam,' Result',(0,p1),font,f2,(0,255,255),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis_result(image0_add_grad_cam)
                    
                    image0_add_grad_cam2=copy.deepcopy(image0)
                    image0_add_grad_cam2 =cv2.putText(image0_add_grad_cam2,' Grad Cam',(0,p1),font,f2,(255,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis_gg(image0_add_grad_cam2)
                    cv2.waitKey(self.time000)
                    
                    image0_add_grad_cam3=copy.deepcopy(image0)
                    image0_add_grad_cam3 =cv2.putText(image0_add_grad_cam3,' 3D Mesh',(0,p1),font,f2,(255,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis_mesh(image0_add_grad_cam3)
                    cv2.waitKey(self.time000)
                    
                    
            # image0_add_mask=copy.deepcopy(image0)
            # image0_add_mask =cv2.putText(image0_add_mask,' image',(0,p1),font,f1,(255,0,255),3)  #text,coordinate,font,size of text,color,thickness of font
            cv2.waitKey(5*self.time000)
            
            self.dis1(image0)        
            
            cv2.imwrite('image0.jpg',image0)
            cv2.imwrite('image01.jpg',image0)        
            cv2.imwrite('out.jpg',image0)
            

        blocks=['loadimage','DICOM_input','cropeimage_2','cropeimage','applysegmentation','applysegmentation','load_models',
        'saveimage','fullscreen1','saveimage_2','fullscreen2','pushButtonTheme','pushButtonColor2','pushButtonLanguage','pushButtonFont','time1','time2','time_p',
        'pushButtonColor1','pushButtonColor2','pushButtonLanguage','pushButtonFont',
        'languageComboBox' 'ThemeComboBox',]
        self.enable_b_function(blocks,'True')
        
        self.ui.segmentation_box.setEnabled(True)
        
        blocks= ['fullscreen1','segmentation_box','applysegmentation',
                  'segmentor1','segmentor2','segmentor3','segmentor4',
                  'segmentor5','segmentor6','segmentor7','segmentor8','load_models']
        self.enable_b_function(blocks,'True')
        
        
    def load_mask_function(self):
        import os
        path0= os.getcwd()
        fname = QFileDialog.getOpenFileName(self, 'Open file', path0)
        # print(fname)
        if fname[0] !='':
            import cv2
            import numpy as np
            MASK=cv2.imread(fname[0])   
            self.mask_e=1
            MASK= cv2.cvtColor(MASK, cv2.COLOR_BGR2GRAY )
            print(np.shape(MASK))
            print(np.unique(MASK))
            MASK3=np.zeros((np.shape(MASK)[0] ,np.shape(MASK)[1] ,3 ))
            # TH=[0,20,40,60]
            i1,j1 = np.where(MASK ==0 )
            i2,j2 = np.where(MASK ==64 )
            i3,j3 = np.where(MASK ==128 )
            i4,j4 = np.where(MASK == 191 )
            MASK3[i2,j2,0] =1
            MASK3[i3,j3,1] =1
            MASK3[i4,j4,2] =1
            # nearst  sssss
            MASK3 = cv2.resize(MASK3,(128,128))
            
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
             
            
            MASK3[:,:,0]=ther(MASK3[:,:,0], np.mean(MASK3[:,:,0]))
            MASK3[:,:,1]=ther(MASK3[:,:,1], np.mean(MASK3[:,:,1]))
            MASK3[:,:,2]=ther(MASK3[:,:,2], np.mean(MASK3[:,:,2]))

            self.mask=MASK3
            print(np.shape(MASK3))
            print(np.shape(self.mask))
            MASK_display = cv2.resize(255*MASK3, (840, 680))
            TEXT=  ' mask'
            
            # cv2.imshow(TEXT,MASK)
            # cv2.waitKey(3000)
            # cv2.destroyWindow(TEXT)
            MASK0=MASK
            try:
                MASK0=(MASK[:,:,0]+MASK[:,:,1]+MASK[:,:,2])/3
                MASK0=int (MASK0)
            except:s=1
            cv2.imwrite('MASK.jpg',MASK)
            cv2.imwrite('MASK1.jpg',MASK)        
            cv2.imwrite('MASK2.jpg',MASK)
            #       stage 2
            self.ui.applysegmentation.setEnabled(True);
            self.ui.load_models.setEnabled(True);
            self.ui.cal_res1.setEnabled(True);
            
            self.ui.result_seg_box.setEnabled(True);
            self.ui.dice_overal.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.cal_res1.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.dice_RV.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.dice_Myo.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.mse1.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")

            
            
            self.ui.status.setText(' Mask is loaded '+ str(np.shape(MASK0)))
            
            
            # np.where(MASK0==3)
            self.dis2(MASK_display )
            
        
        
    def DICOM_input_function(self):
        
        import function_input as u
        import cv2
        import numpy as np
        image0 = u.camera(0)
        self.dis1(image0)
        self.dis2(image0)
        cv2.imwrite('image0.jpg',image0)
        cv2.imwrite('image01.jpg',image0)        
        cv2.imwrite('out.jpg',image0)   
        self.stage=2
        stage =self.stage
        self.stage_function(stage)
#        stage 1 
                
    def cropeimage_function(self):
        # self.ui.time_p.setEnabled(False);
        from subprocess import call
        import cv2
        call(["python", "crop_1.py"])
        image0 = cv2.imread('image01.jpg')     
        self.dis1(image0)
        # self.dis2(image0)
        cv2.imwrite('image0.jpg',image0)
        cv2.imwrite('image01.jpg',image0)        
        cv2.imwrite('out.jpg',image0)  

                
    def cropeimage_2_function(self):
        
         
        import function_input as u
        import cv2
        import numpy as np
        image =cv2.imread('image01.jpg')
        height, width, bytesPerComponent = image.shape
        if height>25:
            if width>25:
                image0=image[10:-10,10:-10,:]
        image01=image0
        self.stage=2
        stage =self.stage
        self.stage_function(stage)
        self.out = image0
        font = cv2.FONT_HERSHEY_COMPLEX
        f1=512 / np.shape(image0) [0]   
        p1=int(np.fix(np.shape(image0) [0] /2  ))
        
        import copy
        image0_add_mask=copy.deepcopy(image0)
        image0_add_mask =cv2.putText(image0_add_mask,' MASK',(0,p1),font,f1,(0,255,0),3)
        image0_add_grad_cam=copy.deepcopy(image0)
        image0_add_grad_cam =cv2.putText(image0_add_grad_cam,' Grad Cam',(0,p1),font,0.8*f1,(0,0,255),3)  #text,coordinate,font,size of text,color,thickness of font
        self.dis_result(image0_add_grad_cam)
        
        image0_add_grad_cam2=copy.deepcopy(image0)
        
        image0_add_grad_cam2 =cv2.putText(image0_add_grad_cam2,' Grad Cam2',(0,p1),font,0.8*f1,(255,0,0),3)  #text,coordinate,font,size of text,color,thickness of font
        
        self.dis_gg(image0_add_grad_cam2)
        self.dis1(image0)
        self.dis2(image0_add_mask)
        
        cv2.imwrite('image0.jpg',image0)
        cv2.imwrite('image01.jpg',image0)        
        cv2.imwrite('out.jpg',image0) 
              
    def saveimage_function(self):
        import os 
        path0= os.getcwd()
        fname = QFileDialog.getSaveFileName(self, 'Open file', path0)
        import cv2
        import numpy as np
        image0=cv2.imread('image0.jpg')
        image01 =  image0
        height = image01.shape[0]
        width = image01.shape[1]
        ratio=680/height
        x11=np.float16(np.fix(width*ratio))
#        print(height)
#        print(width)
#        print(x11)
        fname2='in'
        image01 = cv2.resize(image0,(680,x11))
        cv2.imwrite('image01.jpg',image01)
        cv2.imwrite(fname+'.jpg',image0)
        pixmap1= QtGui.QPixmap()
        self.stage=2
        stage =self.stage
        self.stage_function(stage)

    def saveimage_2_function(self):
        import os 
        path0= os.getcwd()
        fname = QFileDialog.getSaveFileName(self, 'Open file', path0)
        import cv2
        import numpy as np
        image0=cv2.imread('image01.jpg')
        image01 =  image0 
        fname2='out'
        cv2.imwrite(fname+'_out.jpg',image0)
        cv2.imwrite('out.jpg',image01)
        pixmap1= QtGui.QPixmap()
        self.stage=2
        stage =self.stage
        self.stage_function(stage) 
    
# =============================================================================
#   f2
# =============================================================================
    def dispAmount(self):
        self.pre=0
        amount=0;cnt=0
        self.pre=amount    
        if cnt==0:
            cnt2='none'
        if cnt==1:
            cnt2='One Item'
        if cnt==2:
            cnt2='Two Items'
        if cnt==3:
            cnt2='Three Items'
        ##self.ui.label_pre.setText('You Select '+cnt2)
        self.ui.status.setText('You Select '+cnt2)
        
       
    def pre_bar_function(self, value):    
                
        self.ui.label_class.setText(' ')
        import numpy as np
        ##self.ui.label_pre.setText("Level : "+str(np.fix((2.55/0.99)* value)))
        self.th_pre=1+np.fix((2.55/0.99)* value)      

    def zoom_bar_1_function(self, value):    
        import numpy as np
        v=(np.fix (10*(0.04/0.99)* value))/10
        self.ui.label_zoom1.setText("Level : "+str(v))
        self.zoom1=v
        
    def zoom_bar_2_function(self, value):    
        import numpy as np
        v=(np.fix (10*(0.04/0.99)* value))/10
        self.ui.label_zoom2.setText("Level : "+str(v))
        self.zoom2=v
# =============================================================================
# f3        
# =============================================================================
    def dispAmount3(self):
        self.seg=-1
        amount=0;cnt=0
#        print(amount)
        if self.ui.segmentor1.isChecked()==True:
            amount=0;
        if self.ui.segmentor2.isChecked()==True:
            amount=1; 
        if self.ui.segmentor3.isChecked()==True:
            amount=2; 
        if self.ui.segmentor4.isChecked()==True:
            amount=3;
        if self.ui.segmentor5.isChecked()==True:
            amount=4; 
        if self.ui.segmentor6.isChecked()==True:
            amount=5;
        if self.ui.segmentor7.isChecked()==True:
            amount=6;
        if self.ui.segmentor8.isChecked()==True:
            amount=7;    
        
        
        SEGMENTOR_LIST=['CardSegNet','CardUnet ',
                        'MSTGANet','CE-Net',
                        'UNet++','nnUnet ',
                        'UNet','CE-Deeplab v3',
                        ]    
        self.seg=amount  
        segg=SEGMENTOR_LIST[amount]
        
        self.ui.label_pre_2.setText('You Select '+segg)
        self.ui.status.setText(segg +' is selected as segmentor ')
    
    
    def manualsegmentation_function(self):
        #self.ui.label_pre_2.setText('You Press Manual seg..')
        self.ui.status.setText('You Press Manual seg..')
        self.ui.loadimage.setEnabled(False);
        self.ui.DICOM_input.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        
        self.ui.applysegmentation.setEnabled(False);
        self.ui.load_models.setEnabled(False);
        self.ui.applyclassification.setEnabled(False);
        self.ui.saveimage.setEnabled(False);
        self.ui.fullscreen1.setEnabled(False);
        self.ui.saveimage_2.setEnabled(False);
        self.ui.fullscreen2.setEnabled(False);
        self.ui.pushButtonTheme.setEnabled(False);
        self.ui.pushButtonColor2.setEnabled(False);
        self.ui.pushButtonLanguage.setEnabled(False);
        self.ui.pushButtonFont.setEnabled(False);
        self.ui.time1.setEnabled(False);
        self.ui.time2.setEnabled(False);
        # self.ui.time_p.setEnabled(False);
        self.ui.pushButtonColor1.setEnabled(False);
        self.ui.pushButtonColor2.setEnabled(False);
        self.ui.pushButtonLanguage.setEnabled(False);
        self.ui.pushButtonFont.setEnabled(False); 
        self.ui.languageComboBox.setEnabled(False); 
        self.ui.ThemeComboBox.setEnabled(False);
        #self.ui.input_box.setEnabled(False);
        # self.ui.age1.setEnabled(False);
        # self.ui.sex1.setEnabled(False);
        # self.ui.pos1.setEnabled(False); 
        import numpy as np
        import cv2
        image01 = cv2.imread('image01.jpg')
        cv2.imshow('image01',image01)
        th=self.th+1
        for TTT in range(16):
            T=TTT
            while T>15:
                T=TTT-15
            TT='WAIT/wait ('+str(T+1)+').png'
            import cv2
            src1 = cv2.imread('image01.jpg', cv2.IMREAD_COLOR)
            src2 = cv2.imread(TT, cv2.IMREAD_COLOR)
            src1 = cv2.resize(src1, (640, 480)) 
            src2 = cv2.resize(src2, (640, 480))
            src2=1-src2
            dst = cv2.addWeighted(src1, 1.1, src2, 0.3, 0.0)
            self.dis2(dst);cv2.waitKey(20)
        image01= cv2.imread('image01.jpg')
        th=np.uint16(th)
        import function_segmentation as segmi
        out=segmi.thershold_segmentation(image01,th)
        image02=out
        cv2.imwrite('image02.jpg',image02)
        self.stage=4
        stage =self.stage
        self.stage_function(stage)
        self.out=out
        self.out1=out1
        self.image2 = image2
        import abcd_Features as fea
        
        self.ui.age1.setEnabled(False);
        


        self.stage=5
        stage =self.stage
        self.stage_function(stage)
        
#        stage 1 
        self.ui.loadimage.setEnabled(True);
        self.ui.DICOM_input.setEnabled(True);
        self.ui.cropeimage_2.setEnabled(True);
        self.ui.cropeimage.setEnabled(True);
        self.ui.saveimage.setEnabled(True);
        self.ui.fullscreen1.setEnabled(True);
        self.ui.saveimage_2.setEnabled(True);
        self.ui.fullscreen2.setEnabled(True);
        self.ui.pushButtonTheme.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True);
        self.ui.time1.setEnabled(True);
        self.ui.time2.setEnabled(True);        
        # self.ui.time_p.setEnabled(True);        
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);
#        stage 2
                
#       stage 3 
        self.ui.applyclassification.setEnabled(True);
    def load_models_function(self):
        # print('load_models_function  repairing')
        
        self.ui.status.setText( ' loading . . . ')
        self.ui.status.setText( '1 segmentor is loaded')
        self.ui.status.setText( '2 segmentor is loaded')
        self.ui.status.setText( '3 segmentor is loaded')
        self.ui.status.setText( '4 segmentor is loaded')
        self.ui.status.setText( '5 segmentor is loaded')
        self.ui.status.setText( '6 segmentor is loaded')
        self.ui.status.setText( '7 segmentor is loaded')
        self.ui.status.setText( '8 segmentor is loaded')
        s=1
        
    def applysegmentation_function(self):
        self.ui.progressBar_segmentation.setValue(20)
        self.ui.status.setText('You Press segmentation')
        blocks=['loadimage','DICOM_input','cropeimage_2','cropeimage','applysegmentation','applysegmentation','load_models',
        'saveimage','fullscreen1','saveimage_2','fullscreen2','pushButtonTheme','pushButtonColor2','pushButtonLanguage','pushButtonFont','time1','time2','time_p',
        'pushButtonColor1','pushButtonColor2','pushButtonLanguage','pushButtonFont',
        'languageComboBox' 'ThemeComboBox',]
        self.enable_b_function(blocks,'False')
        
        self.stage=4
        stage =self.stage
        self.stage_function(stage)
        import numpy as np
        import cv2
        image01=self.image1
        image01= cv2.imread('image01.jpg')
        for TTT in range(16):
            T=TTT
            while T>15:
                T=TTT-15
            TT='WAIT/wait ('+str(T+1)+').png'
            import cv2
            src1 = cv2.imread('image01.jpg', cv2.IMREAD_COLOR)
            src2 = cv2.imread(TT, cv2.IMREAD_COLOR)
            src1 = cv2.resize(src1, (640, 480)) 
            src2 = cv2.resize(src2, (640, 480))
            src2=1-src2
            dst = cv2.addWeighted(src1, 1.1, src2, 0.3, 0.0)
            self.dis_result(dst);cv2.waitKey(20)
        image01= cv2.imread('image01.jpg')
       
        # th=self.th+1
        import numpy as np
        import cv2 
        # th=20
        
        import function_segmentation as segmi
        
        if self.seg>0:
            self.ui.progressBar_segmentation.setValue(40)
            self.ui.result_seg_box.setEnabled(True);
            self.ui.cal_res1.setEnabled(True);
            self.ui.result_seg_box.setEnabled(True);
            # self.ui.dice_overal.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            # self.ui.cal_res1.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            # self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            # self.ui.dice_RV.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            # self.ui.dice_Myo.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            # self.ui.mse1.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
        
        
        if self.seg>=0:
            self.ui.progressBar_segmentation.setValue(60)
            import function_segmentation as segmi
            
            if self.ui.segmentor1.isChecked()==True:
                from function_deep_dual import function_deep_dual
                ( YY1 , p, superimposed_img1, superimposed_img)  =function_deep_dual(image01,self.mask)
                self.predicted_e=1
            if self.ui.segmentor3.isChecked()==True:
                ( YY1 , p, superimposed_img1, superimposed_img)  =segmi.deep_mstganet_segmentation(image01,self.mask)
                self.predicted_e=1
            self.ui.progressBar_segmentation.setValue(80)
            try:
                print(np.unique(YY1))
                print(np.shape(YY1))
                # print(np.unique(p))
                # YY3=np.zeros((np.shape(YY1)[0] ,np.shape(YY1)[1] ,3 ))
                # i1,j1 = np.where(YY1 ==0 )
                # i2,j2 = np.where(YY1 ==64 )
                # i3,j3 = np.where(YY1 ==128 )
                # i4,j4 = np.where(YY1 == 191 )
                # YY3[i2,j2,0] =1
                # YY3[i3,j3,1] =1
                # YY3[i4,j4,2] =1
                
                
                
                
                self.predicted =YY1 
                YY1= 255*YY1
                
                superimposed_img1= 255*superimposed_img1
                superimposed_img= 255*superimposed_img
                import copy 
                superimposed_img10 = copy.deepcopy(superimposed_img1)
                superimposed_img1[:,:,0] =superimposed_img10[:,:,2]
                superimposed_img1[:,:,1] =superimposed_img10[:,:,1]
                superimposed_img1[:,:,2] =superimposed_img10[:,:,0]
                superimposed_img0 = copy.deepcopy(superimposed_img)
                # RGB  BGR
                # 012  210
                
                superimposed_img[:,:,0] =superimposed_img0[:,:,2]
                superimposed_img[:,:,1] =superimposed_img0[:,:,1]
                superimposed_img[:,:,2] =superimposed_img0[:,:,0]
                # superimposed_img1 = cv2.cvtColor(superimposed_img1, cv2.COLOR_BGR2RGB)
                # superimposed_img  = cv2.cvtColor(superimposed_img , cv2.COLOR_BGR2RGB)
                print(np.shape(superimposed_img1))
                
                cv2.imwrite('0_YY1.png', YY1)
                cv2.imwrite('0_p.png',p)
                cv2.imwrite('0_superimposed_img1.png',superimposed_img1)
                cv2.imwrite('0_superimposed_img.png',superimposed_img)
                
                # repairing saed
                self.time0000=time_phase3
                step_phase31=step_phase3-1
                for i in range(step_phase3):
                    self.dis_result(i/step_phase31*YY1); cv2.waitKey( self.time0000)
                    self.dis_gg(i/step_phase31*superimposed_img); cv2.waitKey( self.time0000)
                    # self.dis_grad2_function(i/step_phase31*superimposed_img); cv2.waitKey( self.time0000)
            except:s=1
            self.ui.progressBar_segmentation.setValue(100)    
        import cv2 
        # import abcd_Features as fea
        image01 = cv2.imread('image01.jpg')
        image02=image01
        cv2.imwrite('out.jpg',image02)
        cv2.imwrite('image02.jpg',image02)
        self.stage=4
        stage =self.stage
        self.stage_function(stage)
        
        import numpy as np
      
        self.stage=5
        stage =self.stage
        self.stage_function(stage)        
        
        
        blocks=['loadimage','DICOM_input','cropeimage_2','cropeimage','applysegmentation','applysegmentation','load_models',
        'saveimage','fullscreen1','saveimage_2','fullscreen2','pushButtonTheme','pushButtonColor2','pushButtonLanguage','pushButtonFont','time1','time2','time_p',
        'pushButtonColor1','pushButtonColor2','pushButtonLanguage','pushButtonFont',
        'languageComboBox' 'ThemeComboBox',]
        self.enable_b_function(blocks,'True')
        self.ui.progressBar_segmentation.setValue(100)
    def cal_res1_function(self):
        print( '  cal_res1_function  (self):')
        
        if self.mask_e> 0  and      self.predicted_e>0:
            import numpy as np
            
            # try:
            Y=self.mask
            YY1=self.predicted
            
            print(np.shape(self.mask))
            print(np.shape(self.predicted))
            
            print(np.unique(self.mask))
            print(np.unique(self.predicted))
            
            print(np.shape(YY1))
            print(np.shape(Y))
            
            print(np.unique(YY1))
            print(np.unique(Y))
            print(10*'k')
            try:
                 
                    
                from metricss import dice_coef
                dice_RV=0.01*int(np.mean(100*dice_coef(self.mask[:,:,0], self.predicted[:,:,0], smooth=1)))
                dice_LV=0.01*int(np.mean(100*dice_coef(self.mask[:,:,2], self.predicted[:,:,2], smooth=1)))
                dice_Myo=0.01*int(np.mean(100*dice_coef(self.mask[:,:,1], self.predicted[:,:,1], smooth=1)))
                B1=1-(self.mask[:,:,0]+self.mask[:,:,1]+self.mask[:,:,2])
                B2=1-(self.predicted[:,:,0]+self.predicted[:,:,1]+self.predicted[:,:,2])
                
                d_b=0.01*int(np.mean(100*dice_coef(B1, B2, smooth=1)))
                dice_overal = ( (dice_RV+dice_LV+dice_Myo+d_b))/4
                mse1=np.mean(np.abs(self.mask[:,:,:]-self.predicted[:,:,:]))
            # dice_RV,dice_LV,dice_Myo,mse1,d_b = function_dice2 (self.mask,self.predicted )
            except:
                dice_RV=-1000
                dice_LV= -1000
                dice_Myo=-1000           
                dice_overal=-10000
                mse1=-1000
            
        print(self.mask_e  , self.predicted_e )    
        if self.mask_e==0  and      self.predicted_e>0:           
            dice_RV=' Need Target mask'
            dice_LV=' Need Target mask'
            dice_Myo=' Need Target mask'
            dice_overal=' Need Target mask'
            mse1=dice_overal
        if self.mask_e>0  and      self.predicted_e==0:           
            dice_RV=' Need predicted mask'
            dice_LV=' Need predicted mask'
            dice_Myo=' Need predicted mask'
            dice_overal=' Need predicted mask'
            mse1=dice_overal
        if self.mask_e==0  and      self.predicted_e==0:           
            dice_RV=' Need p and T mask'
            dice_LV=' Need p and T mask'
            dice_Myo=' Need p and T mask'
            dice_overal=' Need p and T mask'
            mse1=dice_overal
    
        self.ui.dice_overal.setText('DICE Overal ='+str(dice_overal)+'%')
        self.ui.dice_RV.setText(    'DICE RV       ='+str(dice_RV)+'%')
        self.ui.dice_LV.setText(    'DICE LV       ='+str(dice_LV)+'%')
        self.ui.dice_Myo.setText(   'DICE Myo     ='+str(dice_Myo)+'%')
        self.ui.mse1.setText(       'MSE SCORE   ='+str(mse1)+' %')
        
    def segmentationbar_function(self, value):    
        
         
        import numpy as np
        self.th=1+np.fix((2.55/0.99)* value)
        
 
# =============================================================================
#  f5      
# =============================================================================
    def applyclassification_function(self):
        
        self.ui.status.setText('You Press Apply classification')
        blocks=['time2','loadimage',                'cropeimage_2','cropeimage','applysegmentation','applysegmentation','load_models',
        'saveimage','fullscreen1','saveimage_2','fullscreen2','pushButtonTheme','pushButtonColor2','pushButtonLanguage','pushButtonFont','time1','time2','time_p',
        'pushButtonColor1','pushButtonColor2','pushButtonLanguage','pushButtonFont',
        'languageComboBox' 'ThemeComboBox',]
        self.enable_b_function(blocks,'False')
        
# #        self.ui.time1.setEnabled(False)
#         self.ui.time2.setEnabled(False)
        
#         self.ui.loadimage.setEnabled(False);
#         self.ui.DICOM_input.setEnabled(False);
#         self.ui.cropeimage_2.setEnabled(False);
#         self.ui.cropeimage.setEnabled(False);
        
#         self.ui.applysegmentation.setEnabled(False);
#         #self.ui.applyinputdata.setEnabled(False);
#         self.ui.applyclassification.setEnabled(False);
#         self.ui.saveimage.setEnabled(False);
#         self.ui.fullscreen1.setEnabled(False);
#         self.ui.saveimage_2.setEnabled(False);
#         self.ui.fullscreen2.setEnabled(False);
#         self.ui.pushButtonTheme.setEnabled(False);
#         self.ui.pushButtonColor2.setEnabled(False);
#         self.ui.pushButtonLanguage.setEnabled(False);
#         self.ui.pushButtonFont.setEnabled(False);
#         self.ui.time1.setEnabled(False);
#         self.ui.time2.setEnabled(False);
#         # self.ui.time_p.setEnabled(False);
#         self.ui.pushButtonColor1.setEnabled(False);
#         self.ui.pushButtonColor2.setEnabled(False);
#         self.ui.pushButtonLanguage.setEnabled(False);
#         self.ui.pushButtonFont.setEnabled(False); 
#         self.ui.languageComboBox.setEnabled(False); 
#         self.ui.ThemeComboBox.setEnabled(False);         
        # for TTT in range(16):
        #     T=TTT
        #     while T>15:
        #         T=TTT-15
        #     TT='WAIT/wait ('+str(T+1)+').png'
        #     import cv2
        #     src1 = cv2.imread('image01.jpg', cv2.IMREAD_COLOR)
        #     src2 = cv2.imread(TT, cv2.IMREAD_COLOR)
        #     src1 = cv2.resize(src1, (640, 480)) 
        #     src2 = cv2.resize(src2, (640, 480))
        #     src2=1-src2
        #     dst = cv2.addWeighted(src1, 1.1, src2, 0.3, 0.0)
        #     self.dis2(dst);cv2.waitKey(20)
        self.stage=6
        stage =self.stage
        self.stage_function(stage)
        classifer =self.classifer
        for i in range(101):
            cv2.waitKey(100)
            self.ui.progressBar_classification.setValue(i)
        
        for i in range(2):
            blocks=['class_1','class_2','class_3','class_4','class_5',]
            for j in range(25):
                C='rgb('+str(10*j)+',' +str(10*j)+',' +str(10*j)+')'
                self.color_b_function(blocks, C);cv2.waitKey(20)
            for j in range(25):
                C='rgb('+str(255-10*j)+',' +str(255-10*j)+',' +str(255-10*j)+')'
                self.color_b_function(blocks, C);cv2.waitKey(20)
        self.color_b_function(blocks, 'rgb(158, 221, 255);','rgb(0, 0, 0);');    
        for i in range(10):
            self.ui.class_1.setStyleSheet("QWidget { color: rgb(0, 0, 0);background-color: rgb(255, 255, 255); }")
            cv2.waitKey(200)
            self.ui.class_1.setStyleSheet("QWidget { color: rgb(255, 255, 255);background-color: rgb(0, 0, 0); }")
            cv2.waitKey(10)
            self.ui.class_1.setStyleSheet("QWidget { color: rgb(0, 0, 0);background-color: rgb(158, 221, 255); }")
            cv2.waitKey(10)
            self.ui.class_1.setStyleSheet("QWidget { color: rgb(0, 0, 0);background-color: rgb(60, 221, 30); }")
            cv2.waitKey(20)
            self.ui.class_1.setStyleSheet("QWidget { color: rgb(255, 255, 255);background-color: rgb(255, 0, 0); }")
            cv2.waitKey(200)
        # a=255
        # b=255 
        # c=255 
        
        blocks=['time2','loadimage',                'cropeimage_2','cropeimage','applysegmentation','applysegmentation','load_models',
        'saveimage','fullscreen1','saveimage_2','fullscreen2','pushButtonTheme','pushButtonColor2','pushButtonLanguage','pushButtonFont','time1','time2','time_p',
        'pushButtonColor1','pushButtonColor2','pushButtonLanguage','pushButtonFont',
        'languageComboBox' 'ThemeComboBox',]
        self.enable_b_function(blocks,'True')
        
# =============================================================================
#  f6 classification       
# =============================================================================
    def dispAmount2(self):
        amount2=0
        if self.ui.classifier1.isChecked()==True:
            amount2=1
        if self.ui.classifier2.isChecked()==True:
            amount2=2
        if self.ui.classifier3.isChecked()==True:
            amount2=3
 
        if amount2==0:
            classifer='none'
        if amount2==1:
            classifer='Deep learning'

        if amount2==2:
            classifer='PWC'
        if amount2==3:
            classifer='classifier3 on Deep Features'
        
        self.classifer=classifer
        #self.ui.label_class.setText('You Select '+classifer)
        self.ui.status.setText('You Select '+classifer)
# =============================================================================
# f7 result
# =============================================================================


# =============================================================================
# f8 below figure
# =============================================================================
    def fullscreen1_function(self):
        import cv2
        img = cv2.imread('image0.jpg')
  
        scale_percent = 100*self.zoom1 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
#        dim2 = (640, 480)
        # resize image
        image0 = cv2.resize(img, dim , interpolation = cv2.INTER_AREA) 
        cv2.imshow('Original image',image0)
        cv2.destroyWindow('Original image')
        cv2.imshow('Original image',image0)
        
        
    def fullscreen2_function(self):
        import cv2
        img = cv2.imread('out.jpg')
        scale_percent = 100*self.zoom2 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
#        dim2 = (640, 480)
        # resize image
        result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
         
        cv2.imshow('result image',result)
        cv2.destroyWindow('result image')
        cv2.imshow('result image',result)

    def Author_function(self):
        if self.Author==0:
            import cv2
            image01 = cv2.imread('images/arm.jpg')
            # image01 = cv2.resize(image00,(10,300))
            ResultImage = cv2.cvtColor(image01, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = ResultImage.shape
            height=180
            width=250
            
            bytesPerLine = 3 * width
            QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(QImg)
            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(pixmap)
            self.ui.fig3.setScene(scene)
            image01 = cv2.imread('images/Hamed.jpg')

            ResultImage = cv2.cvtColor(image01, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = ResultImage.shape
            bytesPerLine = 3 * width
            QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(QImg)
            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(pixmap)
            self.ui.fig4.setScene(scene)
            self.Author=self.Author+1
            image01 = cv2.imread('images/Hamed (3).jpg')
            cv2.imshow('Hamed image',image01)
            cv2.waitKey(1000)
            cv2.destroyWindow('Hamed image')           
        else:
            import cv2
            image01 = cv2.imread('images/banner 3.jpg')
            ResultImage = cv2.cvtColor(image01, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = ResultImage.shape
            bytesPerLine = 3 * width
            QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(QImg)
            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(pixmap)
            self.ui.fig3.setScene(scene)
            image01 = cv2.imread('images/banner 3.jpg')
            cv2.imshow('Hamed image',image01)
            cv2.waitKey(1000)
            cv2.destroyWindow('Hamed image')
            self.Author=self.Author+1
        
    def mask_function(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(80,79))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig2.setScene(scene)
        
    # def dis_grad2_function(self,image0):
    #     import cv2
    #     import numpy as np
    #     x0 = np.size(image0,0);
    #     x1 = np.size(image0,1)
    #     resized = cv2.resize(image0,(350,300))
    #     ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
    #     height, width, bytesPerComponent = ResultImage.shape
    #     bytesPerLine = 3 * width
    #     QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    #     pixmap = QtGui.QPixmap.fromImage(QImg)
    #     scene = QtWidgets.QGraphicsScene()
    #     scene.addPixmap(pixmap)
    #     self.ui.fig_grad2.setScene(scene)   
    def dis_mesh(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(350,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_mesh.setScene(scene)
        
    def dis1(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(350,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig1.setScene(scene)
        
        
        
    def dis2(self,image0):
        import numpy as np
        # print(type(image0),np.max(image0) , np.min(image0) )
        import cv2
        import numpy as np
        self.counter2=self.counter2 + 1 
        import os
        try:
            os.mkdir('temp')
        except:s=1
        path_0='temp//mask'+str(self.counter2)+'.png'
        cv2.imwrite(path_0,image0)
        image0 = cv2.imread(path_0)
        
        resized = cv2.resize(image0,(350,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig2.setScene(scene)        

    def dis_arm(self,image0):
        import cv2
        # resized = cv2.resize(image0,(450,300))
        ResultImage = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig3.setScene(scene)
    def dis_flag(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(95,90))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_flag.setScene(scene) 
        
    def dis_result(self,image0):
        import cv2
        import numpy as np
        self.counter1=self.counter1 + 1 
        import os
        try:
            os.mkdir('temp')
        except:s=1
        
        path_0='temp//grad'+str(self.counter1)+'.png'
        cv2.imwrite(path_0,image0)
        image0 = cv2.imread(path_0)
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(350,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_result.setScene(scene)
        
    def dis_gg(self,image0):
        import cv2
        import numpy as np
        self.counter1=self.counter1 + 1 
        import os
        try:
            os.mkdir('temp')
        except:s=1
        path_0='temp//grad2_'+str(self.counter1)+'.png'
        cv2.imwrite(path_0,image0)
        image0 = cv2.imread(path_0)
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(200,200))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_gg.setScene(scene)
    # def dis_logo(self,image0):
    #     import cv2
    #     import numpy as np
    #     x0 = np.size(image0,0);
    #     x1 = np.size(image0,1)
    #     resized = cv2.resize(image0,(140-5,190-5))
    #     # resized =   image0 
    #     ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
    #     height, width, bytesPerComponent = ResultImage.shape
    #     bytesPerLine = 3 * width
    #     QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    #     pixmap = QtGui.QPixmap.fromImage(QImg)
    #     scene = QtWidgets.QGraphicsScene()
    #     scene.addPixmap(pixmap)
    #     self.ui.fig_logo.setScene(scene)
        
        
    def dis_author(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(450,300))
        resized =  image0 
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig4.setScene(scene)  
        
        
        
    def saed(self,image,flage):
        self.ui.fig1.setEnabled(True)
        self.ui.fig2.setEnabled(False)
        self.ui.fig3.setEnabled(False)
        self.ui.fig4.setEnabled(False)
    def changeFont_function(self):
        
        myFont=self.myfont  #QtGui.QFont(self.ui.fontComboBox.itemText(self.ui.fontComboBox.currentIndex()),15)
        self.ui.textEdit.setFont(myFont)



    def language_function(self,languager):
        cnt=0
        self.ui.loadimage.setText(languager[cnt]);cnt=cnt+1
        self.ui.DICOM_input.setText(languager[cnt]);cnt=cnt+1
        self.ui.load_mask.setText(languager[cnt]);cnt=cnt+1
        self.ui.loadimage_2.setText(languager[cnt]);cnt=cnt+1
        self.ui.cropeimage_2.setText(languager[cnt]);cnt=cnt+1
        self.ui.cropeimage.setText(languager[cnt]);cnt=cnt+1
        self.ui.applysegmentation.setText(languager[cnt]);cnt=cnt+1                       
        self.ui.load_models.setText(languager[cnt]);cnt=cnt+1                       
        self.ui.classifier1.setText(languager[cnt]);cnt=cnt+1
        self.ui.classifier2.setText(languager[cnt]);cnt=cnt+1
        self.ui.classifier3.setText(languager[cnt]);cnt=cnt+1
        self.ui.applyclassification.setText(languager[cnt]);cnt=cnt+1             
        self.ui.original_image.setText(languager[cnt]);cnt=cnt+1             
        self.ui.result_image.setText(languager[cnt]);cnt=cnt+1             
        self.ui.saveimage.setText(languager[cnt]);cnt=cnt+1
        self.ui.fullscreen1.setText(languager[cnt]);cnt=cnt+1
        self.ui.saveimage_2.setText(languager[cnt]);cnt=cnt+1
        self.ui.fullscreen2.setText(languager[cnt]);cnt=cnt+1 
        self.ui.Author.setText(languager[cnt]);cnt=cnt+1
        self.ui.pushButtonColor1.setText(languager[cnt]);cnt=cnt+1 
        # self.ui.color111.setText(languager[cnt]);cnt=cnt+1 
        self.ui.pushButtonColor2.setText(languager[cnt]);cnt=cnt+1 
        self.ui.pushButtonFont.setText(languager[cnt]);cnt=cnt+1 

        self.ui.class_1.setText(languager[cnt]);cnt=cnt+1 
        self.ui.class_2.setText(languager[cnt]);cnt=cnt+1 
        self.ui.class_3.setText(languager[cnt]);cnt=cnt+1 
        self.ui.class_4.setText(languager[cnt]);cnt=cnt+1 
        self.ui.class_5.setText(languager[cnt]);cnt=cnt+1 
        
        
        from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
        from PyQt5.QtGui import QFont 
        
        font = QFont()
        font.setFamily("B Nazanin")
        
        self.ui.loadimage.setFont(font); 
        self.ui.DICOM_input. setFont(font); 
        self.ui.load_mask. setFont(font); 
        self.ui.loadimage_2. setFont(font); 
        self.ui.cropeimage_2. setFont(font); 
        self.ui.cropeimage. setFont(font); 
        self.ui.applysegmentation. setFont(font);                        
        self.ui.load_models. setFont(font);                        
        self.ui.classifier1. setFont(font); 
        self.ui.classifier2. setFont(font); 
        self.ui.classifier3. setFont(font); 
        self.ui.applyclassification. setFont(font);              
        self.ui.original_image. setFont(font);              
        self.ui.result_image. setFont(font);              
        self.ui.saveimage. setFont(font); 
        self.ui.fullscreen1. setFont(font); 
        self.ui.saveimage_2. setFont(font); 
        self.ui.fullscreen2. setFont(font);  
        self.ui.Author. setFont(font); 
        self.ui.pushButtonColor1. setFont(font);  
        # self.ui.color111. setFont(font);  
        self.ui.pushButtonColor2. setFont(font);  
        self.ui.pushButtonFont. setFont(font);  
        
        self.ui.class_1. setFont(font);  
        self.ui.class_2. setFont(font);  
        self.ui.class_3. setFont(font);  
        self.ui.class_4. setFont(font);  
        self.ui.class_5. setFont(font);
    
    def color_b_function(self,blocks,color,color1='rgb(0, 0, 0)'):
        for i in blocks:
            T='self.ui.'+i+'.setStyleSheet("QWidget { color:'+color1+'; background-color:'+color+'}")'
            try:
                eval(T)
            except: print(i,' is not saed')
    
    def enable_b_function(self,blocks,enable):
        for i in blocks:
            T='self.ui.'+i+'.setEnabled('+enable+') '
            self.ui.segmentor1.setEnabled(True)
            try:
                eval(T)             
            except: 
                print(i,' is not saed')
                print(T)
    def stage_function(self,stage):
        print('stage_function',stage)
        
        if self.stage >=1:
            blocks= ['fullscreen1','segmentation_box','applysegmentation',
                      'segmentor1','segmentor2','segmentor3','segmentor4',
                      'segmentor5','segmentor6','segmentor7','segmentor8','load_models']
            self.enable_b_function(blocks,'True') 
               
         
            
        if self.stage >1:
            
            
            self.ui.classification_box.setEnabled(True)
            blocks= ['classifier1','classifier2','classifier3',
                      'label_class','applyclassification','result_box','time1',
                      'time2' ]
            self.enable_b_function(blocks,'True')
             
              
# =============================================================================
#             end2222222222222222222
# =============================================================================
if __name__=="__main__":    
    app = QApplication(sys.argv)
    myapp = MyForm()
    # myapp.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    myapp.show()
    sys.exit(app.exec_())
    
    # from PyQt5 import QtCore
    # create window here...
    # window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

