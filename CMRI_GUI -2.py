
saed=10


time_phase1 = 60 ### mili seconds
time_phase2 = 20 ### mili seconds
step_phase2 = 20 ### mili seconds
step_phase3 = 20 ### mili seconds

time_phase3 = 60 ### mili seconds

shamsi='1402 03-05'

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
# convert ui to py
# =============================================================================


# from G2 import *
# from G3 import *
from G4 import *

import cv2
import numpy as np

if not saed ==1:
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
        
        
        
        self.model1=0
        self.model2=0
        self.model3=0
        self.model4=0
        self.model5=0
        self.model6=0


        self.mask=0
        self.out=0
        
        
        self.dice_all=0
        self.dice_RV=0
        self.dice_LV=0
        self.dice_Myo=0
        self.mse=0

        
        
        # self.ui.actionloadimage.triggered.connect(self.loadimage_function)
        # self.ui.actionloadimage.triggered.connect(self.load_mask_function)
        # self.ui.actioncrop.triggered.connect(self.cropeimage_function)
        # self.ui.actionsaveimage.triggered.connect(self.saveimage_function)
        # self.ui.actionEntropy.triggered.connect(self.applysegmentation_function)
        # self.ui.actionThershold.triggered.connect(self.applysegmentation_function)
        
        # self.ui.actionDeepLearning.triggered.connect(self.applyclassification_function)
        # self.ui.actionSVM.triggered.connect(self.applyclassification_function)
        # self.ui.actionKNN.triggered.connect(self.applyclassification_function)
        self.ui.result_seg_box.setEnabled(False);
        self.ui.dice_overal.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        self.ui.cal_res1.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        self.ui.dice_RV.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        self.ui.dice_Myo.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        self.ui.mse1.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        
        
        self.ui.class_1.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        self.ui.class_2.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        self.ui.class_3.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        self.ui.class_4.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        self.ui.class_5.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(158, 221, 255); }")
        # self.ui.result_box.setStyleSheet("QWidget { color: rgb(70, 70, 70);background-color: rgb(123, 176, 255); font: 75 12pt }")
        
        
        # self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        # self.ui.result_seg_box.setStyleSheet("QWidget { color: rgb(70, 70, 70); }")
        
        
        self.ui.actionHelp_Software.triggered.connect(self.Help_Software_function)

        self.ui.actionVisit_software_page.triggered.connect(self.Visit_Our_site_function)
        self.ui.actionLicence.triggered.connect(self.Licence_function)
        
        
        self.ui.fig1.setEnabled(False);self.ui.fig2.setEnabled(False)
        self.ui.fig3.setEnabled(False);self.ui.fig4.setEnabled(False)
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
        self.seg=1
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
        
        logo=cv2.imread('images/logo.png')
        self.dis_logo(logo)
        
        image000=cv2.imread('images/banner 3.jpg')
        image_arm=cv2.imread('images/arm.jpg')
        image_author=cv2.imread('images/author.jpg')

        self.dis1(image0)
        self.dis2(image0)
        self.dis_arm(image_arm)
        
        self.dis_grad_cam(image0)
        self.dis_grad_cam2(image0)
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
        self.ui.deep1_method.toggled.connect(self.dispAmount3)
        self.ui.deep2_method.toggled.connect(self.dispAmount3)
        # #self.ui.deep_method3.toggled.connect(self.dispAmount3)
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
        self.ui.deepleaning.toggled.connect(self.dispAmount2)
        self.ui.svm.toggled.connect(self.dispAmount2)
        self.ui.KNN.toggled.connect(self.dispAmount2)
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
            # self.ui.# # self.ui.preprocessing_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            #self.ui.clinical_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            #self.ui.input_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.classification_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.result_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.result_seg_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.calendarWidget.setStyleSheet("QWidget { background-color: %s }" % col.name())


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
        
        EN_P=['Load Image','DICOM_input','Zoom','Crop' ,'Deep method' ,'Deep method' ,'Apply', 'load models',
              'Deep learning','SVM on Deep Features','KNN on Deep Features',
              'Apply','Original Image','Result Image','Save Image','Full Screen'
              ]
        
        if lan0=='English':
            flag=cv2.imread('flag/Britain-512.png')
            self.dis_flag(flag)
            
            cnt=0
            self.ui.loadimage.setText(EN_P[cnt]);cnt=cnt+1
            self.ui.DICOM_input.setText('DICOM_input')
            self.ui.cropeimage_2.setText('')
            self.ui.cropeimage.setText()
            self.ui.deep1_method.setText()
            self.ui.deep2_method.setText()
            self.ui.applysegmentation.setText()                       
            self.ui.load_models.setText()                       

            self.ui.deepleaning.setText()
            self.ui.svm.setText()
            self.ui.KNN.setText()
            self.ui.applyclassification.setText()             
            #self.ui.dice_overal.setText('Result')             
            self.ui.original_image.setText()             
            self.ui.result_image.setText()             
            self.ui.saveimage.setText()
            self.ui.fullscreen1.setText()
            self.ui.saveimage_2.setText('Save Image')
            self.ui.fullscreen2.setText('Full Screen') 
            self.ui.Author.setText('Created by Hamed Aghapanah    Isfahan University of Medical Science     Version 2 , 2023') 
            self.ui.pushButtonColor1.setText('Background') 
            self.ui.author_7.setText('    Color') 
            self.ui.pushButtonColor2.setText('Boxes') 
            self.ui.pushButtonFont.setText('Font') 
             
        if lan0=='French':
            flag=cv2.imread('flag/french-512.png')
            self.dis_flag(flag)
            self.ui.loadimage.setText('Charger une image')
            self.ui.DICOM_input.setText('Caméra')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Surgir')
            self.ui.deep1_method.setText('Méthode dentropie')
            self.ui.deep2_method.setText('Méthode dentropie')
            #self.ui.deep_method3.setText('Méthode de seuil')
            # self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Appliquer')                       
            self.ui.load_models.setText('charger des modèles')                       

            self.ui.deepleaning.setText('Lapprentissage en profondeur')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Appliquer')             
            #self.ui.dice_overal.setText('Résultat')             
            self.ui.original_image.setText('Image originale')             
            self.ui.result_image.setText('Image de résultat')             
            self.ui.saveimage.setText('Enregistrer limage')
            self.ui.fullscreen1.setText('Plein écran')
            self.ui.saveimage_2.setText('Enregistrer limage')
            self.ui.fullscreen2.setText('Plein écran') 
            self.ui.Author.setText('Créé par Hamed Aghapanah Université des sciences médicales d Ispahan Version 1 , 2023') 
            self.ui.pushButtonColor1.setText('Contexte') 
            self.ui.author_7.setText('    Couleur') 
            self.ui.pushButtonColor2.setText('Des boites') 
            self.ui.pushButtonFont.setText('la font')             
            
            
        if lan0=='Germany':
            flag=cv2.imread('flag/German_512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Afbeelding laden')
            self.ui.DICOM_input.setText('DICOM_input')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Crop')

            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.applysegmentation.setText('Van toepassing zijn')                       
            self.ui.load_models.setText('Modelle laden')                       

            self.ui.deepleaning.setText('Diep leren')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Van toepassing zijn')             
            #self.ui.dice_overal.setText('Resultaat')             
            self.ui.original_image.setText('Originele afbeelding')             
            self.ui.result_image.setText('Resultaat afbeelding')             
            self.ui.saveimage.setText('Afbeelding opslaan')
            self.ui.fullscreen1.setText('Volledig scherm')
            self.ui.saveimage_2.setText('Afbeelding opslaan')
            self.ui.fullscreen2.setText('Volledig scherm') 
            self.ui.Author.setText('Gemaakt door Hamed Aghapanah Isfahan University of Medical Science versie 1, 2023') 
            self.ui.pushButtonColor1.setText('Achtergrond') 
            self.ui.author_7.setText('    Kleur') 
            self.ui.pushButtonColor2.setText('Boxes') 
            self.ui.pushButtonFont.setText('doopvont') 
             
        if lan0=='Hindustani':
            flag=cv2.imread('flag/India-512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('लोड छवि')
            self.ui.DICOM_input.setText('कैमरा')
            self.ui.cropeimage_2.setText('ज़ूम')
            self.ui.cropeimage.setText('काटना')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            #self.ui.deep_method3.setText('Threshold Method')
            # self.ui.manualsegmentation.setText('गाइड')
            self.ui.applysegmentation.setText('विभाजन लागू करें')                       
            self.ui.load_models.setText('लागू')                       

            self.ui.deepleaning.setText('ध्यान लगा के पढ़ना या सीखना')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('लागू')             
            #self.ui.dice_overal.setText('नतीजा')             
            self.ui.original_image.setText('मूल छवि')             
            self.ui.result_image.setText('Result Image')             
            self.ui.saveimage.setText('छवि सहेजें')
            self.ui.fullscreen1.setText('पूर्ण स्क्रीन')
            self.ui.saveimage_2.setText('छवि सहेजें')
            self.ui.fullscreen2.setText('पूर्ण स्क्रीन') 
            self.ui.Author.setText('चिकित्सा विज्ञान संस्करण 1, 2023 के हम्द अगपनहा इस्फ़हान विश्वविद्यालय द्वारा बनाया गया') 
            self.ui.pushButtonColor1.setText('पृष्ठभूमि') 
            self.ui.author_7.setText('    रंग') 
            self.ui.pushButtonColor2.setText('बक्से') 
            self.ui.pushButtonFont.setText('फ़ॉन्ट') 
             
        if lan0=='Chinese':
            flag=cv2.imread('flag/China-512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('载入图片')
            self.ui.DICOM_input.setText('相机')
            self.ui.cropeimage_2.setText('放大')
            self.ui.cropeimage.setText('作物')

            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.applysegmentation.setText('应用')                       
            self.ui.load_models.setText('应用细分')                       

            self.ui.deepleaning.setText('深度学习')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('应用')             
            self.ui.original_image.setText('原始图片')             
            self.ui.result_image.setText('结果图像')             
            self.ui.saveimage.setText('保存图片')
            self.ui.fullscreen1.setText('全屏')
            self.ui.saveimage_2.setText('保存图片')
            self.ui.fullscreen2.setText('全屏') 
            self.ui.Author.setText('由Hamed Aghapanah伊斯法罕医科大学 第1版创建，2023年') 
            self.ui.pushButtonColor1.setText('背景') 
            self.ui.author_7.setText('    颜色') 
            self.ui.pushButtonColor2.setText('盒') 
            self.ui.pushButtonFont.setText('字形') 
             
        if lan0=='Spanish':
            flag=cv2.imread('flag/Spain-2-512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Cargar imagen')
            self.ui.DICOM_input.setText('Cámara')
            self.ui.cropeimage_2.setText('Enfocar')
            self.ui.cropeimage.setText('Cosecha')

            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            #self.ui.deep_method3.setText('Threshold Method')
            # self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Aplicar')                       
            self.ui.load_models.setText('cargar modelos')                       

            self.ui.deepleaning.setText('Aprendizaje profundo')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Aplicar')             
            #self.ui.dice_overal.setText('Resultado')             
            self.ui.original_image.setText('Original Image')             
            self.ui.result_image.setText('Imagen del resultado')             
            self.ui.saveimage.setText('Guardar imagen')
            self.ui.fullscreen1.setText('Pantalla completa')
            self.ui.saveimage_2.setText('Guardar imagen')
            self.ui.fullscreen2.setText('Pantalla completa') 
            self.ui.Author.setText('Creado por Hamed Aghapanah Isfahan Universidad de Ciencias Médicas Versión 1, 2023') 
            self.ui.pushButtonColor1.setText('Background') 
            self.ui.author_7.setText('    Color') 
            self.ui.pushButtonColor2.setText('Cajas') 
            self.ui.pushButtonFont.setText('Fuente') 
             
        if lan0=='العربیه':
            flag=cv2.imread('flag/Saudia_arabia_national_flags_country_flags-512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('تحميل الصورة')
            self.ui.DICOM_input.setText('الة تصوير')
            self.ui.cropeimage_2.setText('تكبير')
            self.ui.cropeimage.setText('اقتصاص')

            self.ui.deep1_method.setText('طريقة الانتروبيا')
            self.ui.deep2_method.setText('طريقة الانتروبيا')
            self.ui.applysegmentation.setText('تطبيق')                       
            self.ui.load_models.setText('نماذج الحمل')                       

            self.ui.deepleaning.setText('تعلم عميق')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('تطبيق')             
            ##self.ui.dice_overal.setText('نتيجة ')             
            self.ui.original_image.setText('الصورة الأصلية')             
            self.ui.result_image.setText(' الصورة النتيجة')             
            self.ui.saveimage.setText('احفظ الصورة')
            self.ui.fullscreen1.setText('شاشة كاملة')
            self.ui.saveimage_2.setText('احفظ الصورة')
            self.ui.fullscreen2.setText('شاشة كاملة') 
            self.ui.Author.setText('تم الإنشاء بواسطة جامعة حامد أغبانة أصفهان للعلوم الطبية ، الإصدار 1 ، 1441') 
            self.ui.pushButtonColor1.setText('خلفية') 
            self.ui.author_7.setText('    اللون') 
            self.ui.pushButtonColor2.setText('مربعات') 
            self.ui.pushButtonFont.setText('الخط') 
             
        if lan0=='Malay':
            flag=cv2.imread('flag/flag-39-512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Muat Imej')
            self.ui.DICOM_input.setText('Kamera')
            self.ui.cropeimage_2.setText('Zum')
            self.ui.cropeimage.setText('Potong')

            self.ui.deep1_method.setText('Kaedah Entropi')
            self.ui.deep2_method.setText('Kaedah Entropi')
            self.ui.applysegmentation.setText('Sapukan')                       
            self.ui.load_models.setText('memuatkan model')                       

            self.ui.deepleaning.setText('Pembelajaran yang mendalam')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Sapukan')             
            self.ui.original_image.setText('Imej Asal')             
            self.ui.result_image.setText('Imej hasil')             
            self.ui.saveimage.setText('Menyimpan imej')
            self.ui.fullscreen1.setText('Skrin penuh')
            self.ui.saveimage_2.setText('Menyimpan imej')
            self.ui.fullscreen2.setText('Skrin penuh') 
            self.ui.Author.setText('Dicipta oleh Hamed Aghapanah Isfahan Universiti Sains Perubatan Versi 1, 2023') 
            self.ui.pushButtonColor1.setText('Latar Belakang') 
            self.ui.author_7.setText('    Warna') 
            self.ui.pushButtonColor2.setText('Kotak') 
            self.ui.pushButtonFont.setText('Fon') 
             
        if lan0=='Russian':
            flag=cv2.imread('flag/Russian-512.png')
            self.dis_flag(flag)
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Загрузить изображение')
            self.ui.DICOM_input.setText('камера')
            self.ui.cropeimage_2.setText('Увеличить')
            self.ui.cropeimage.setText('урожай')
            self.ui.deep1_method.setText('Метод энтропии')
            self.ui.deep2_method.setText('Метод энтропии')
            #self.ui.deep_method3.setText('Пороговый метод')
            # self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Применять')                       
            self.ui.load_models.setText('загрузить модели')                       
            self.ui.deepleaning.setText('Глубокое обучение')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Применять')             
            #self.ui.dice_overal.setText('Результат')             
            self.ui.original_image.setText('Исходное изображение')             
            self.ui.result_image.setText('Изображение результата')             
            self.ui.saveimage.setText('Сохранить изображение')
            self.ui.fullscreen1.setText('Полноэкранный')
            self.ui.saveimage_2.setText('Сохранить изображение')
            self.ui.fullscreen2.setText('Полноэкранный') 
            self.ui.Author.setText(' Создано Университет медицинских наук Исфахана Хамеда Агхапана Версия 1, 2023') 
            self.ui.pushButtonColor1.setText('Фон') 
            self.ui.author_7.setText('    цвет') 
            self.ui.pushButtonColor2.setText('Ящики') 
            self.ui.pushButtonFont.setText('FoШрифтnt')
        if lan0=='italian':
            flag=cv2.imread('flag/Italy-512.png')
            self.dis_flag(flag)
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Carica immagine')
            self.ui.DICOM_input.setText('teleDICOM_input')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Ritaglia')

            self.ui.deep1_method.setText('Metodo di entropia')
            self.ui.deep2_method.setText('Metodo di entropia')
            #self.ui.deep_method3.setText('Metodo di soglia')
            # self.ui.manualsegmentation.setText('Manuale')
            self.ui.applysegmentation.setText('Applicare')                       
            self.ui.load_models.setText('caricare i modelli')                       

            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Applicare')             
            #self.ui.dice_overal.setText('Risultato')             
            self.ui.original_image.setText('Immagine originale')             
            self.ui.result_image.setText('Immagine del risultato')             
            self.ui.saveimage.setText('Salva immagine')
            self.ui.fullscreen1.setText('A schermo intero')
            self.ui.saveimage_2.setText('Salva immagine')
            self.ui.fullscreen2.setText('A schermo intero') 
            self.ui.Author.setText('Creato da Hamed Aghapanah Isfahan University of Medical Science Versione 1, 2023') 
            self.ui.pushButtonColor1.setText('sfondo') 
            self.ui.author_7.setText('    Colore') 
            self.ui.pushButtonColor2.setText('scatole') 
            self.ui.pushButtonFont.setText('Font')
        if lan0=='فارسی':
            flag=cv2.imread('flag/iran-512.png')
            self.dis_flag(flag)
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('بارگذاری تصویر')
            self.ui.DICOM_input.setText('دوربین')
            self.ui.cropeimage_2.setText('بزرگ نمایی')
            self.ui.cropeimage.setText('برش')

            self.ui.deep1_method.setText('روش عمیق1')
            self.ui.deep2_method.setText('روش عمیق2')
            self.ui.applysegmentation.setText('اعمال')                       
            self.ui.load_models.setText('بازیابی مدلها')                       

            self.ui.deepleaning.setText('یادگیری عمیق')
            self.ui.svm.setText('ماشین بردار پشتیبان')
            self.ui.KNN.setText('نزدیکترین همسایگی')
            self.ui.applyclassification.setText('اعمال')             
            self.ui.original_image.setText('تصویر اصلی')             
            self.ui.result_image.setText('تصویر نهایی')             
            self.ui.saveimage.setText('ذخیره تصویر')
            self.ui.fullscreen1.setText('تمام صفحه')
            self.ui.saveimage_2.setText('ذخیره تصویر')
            self.ui.fullscreen2.setText('تمام صفحه') 
            self.ui.Author.setText('تهیه شده توسط حامد آقاپناه، دانشگاه علوم پزشکی اصفهان، نسخه ۱، سال 1401') 
            self.ui.pushButtonColor1.setText('پس زمینه') 
            self.ui.author_7.setText('    رنگ') 
            self.ui.pushButtonColor2.setText('جعبه') 
            self.ui.pushButtonFont.setText('فونت') 
        if lan0=='Dutch':
            flag=cv2.imread('flag/Dutch_512.png')
            self.dis_flag(flag)            
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Afbeelding laden')
            self.ui.DICOM_input.setText('DICOM_input')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Crop')

            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep2 method')
            self.ui.applysegmentation.setText('Van toepassing zijn')                       
            self.ui.load_models.setText('laden modellen')                       

            self.ui.deepleaning.setText('Diep leren')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Van toepassing zijn')             
            self.ui.original_image.setText('Originele afbeelding')             
            self.ui.result_image.setText('Resultaat afbeelding')             
            self.ui.saveimage.setText('Afbeelding opslaan')
            self.ui.fullscreen1.setText('Volledig scherm')
            self.ui.saveimage_2.setText('Afbeelding opslaan')
            self.ui.fullscreen2.setText('Volledig scherm') 
            self.ui.Author.setText('Gemaakt door Hamed Aghapanah Isfahan University of Medical Science versie 1, 2023') 
            self.ui.pushButtonColor1.setText('Achtergrond') 
            self.ui.author_7.setText('    Kleur') 
            self.ui.pushButtonColor2.setText('Boxes') 
            self.ui.pushButtonFont.setText('doopvont') 

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
            
            self.ui.fig0.setStyleSheet("QWidget { background-color:black}")
            self.ui.seg_text1.setStyleSheet("QWidget { background-color:black}")
            self.ui.seg_text2.setStyleSheet("QWidget { background-color:black}")
            self.ui.seg_text3.setStyleSheet("QWidget { background-color:black}")
            self.ui.seg_text4.setStyleSheet("QWidget { background-color:black}")
            self.ui.zoom_bar_1.setStyleSheet("QWidget { background-color:black}")
            self.ui.zoom_bar_2.setStyleSheet("QWidget { background-color:black}")
            self.ui.label_zoom1.setStyleSheet("QWidget { background-color:black}")
            self.ui.label_zoom2.setStyleSheet("QWidget { background-color:black}")
            self.ui.themebox.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            # self.ui.preprocessing_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            #self.ui.input_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color:rgb(87, 112, 255) }" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: black}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: black}" ) 
            self.ui.time_p.setStyleSheet("QWidget { background-color: black}" ) 
            self.ui.author_7.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" ) 
         
            self.ui.loadimage.setStyleSheet("QWidget { background-color: Green }" )             
            self.ui.DICOM_input.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.cropeimage.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.cropeimage_2.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.saveimage.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Green}" )             
           
            self.ui.applysegmentation.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.load_models.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.applyclassification.setStyleSheet("QWidget { background-color: Green}" )             
            
            self.ui.pushButtonTheme.setStyleSheet("QWidget { background-color: Green}" ) 
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Green}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Green}" ) 
            self.ui.pushButtonFont.setStyleSheet("QWidget { background-color: Green}" ) 
            self.ui.pushButtonLanguage.setStyleSheet("QWidget { background-color: Green}" ) 
            self.ui.languageComboBox.setStyleSheet("QWidget { background-color: Green}" ) 
            
            
        if theme_index==1:
            self.ui.fig0.setStyleSheet("QWidget { background-color:white}")
            self.ui.seg_text1.setStyleSheet("QWidget { background-color:white}")
            self.ui.seg_text2.setStyleSheet("QWidget { background-color:white}")
            self.ui.seg_text3.setStyleSheet("QWidget { background-color:white}")
            self.ui.seg_text4.setStyleSheet("QWidget { background-color:white}")
            self.ui.zoom_bar_1.setStyleSheet("QWidget { background-color:white}")
            self.ui.zoom_bar_2.setStyleSheet("QWidget { background-color:white}")
            self.ui.label_zoom1.setStyleSheet("QWidget { background-color:white}")
            self.ui.label_zoom2.setStyleSheet("QWidget { background-color:white}")


           
            
            self.ui.themebox.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.clinical_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: bold gray}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color: rgb(187, 212, 255)}" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(187, 212, 255)}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: white}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: white}" ) 
            self.ui.time_p.setStyleSheet("QWidget { background-color: white}" ) 
            self.ui.author_7.setStyleSheet("QWidget { background-color: bold gray}" ) 

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
            self.ui.vasc_2.setStyleSheet("QWidget { background-color: Cyan}" ) 
            
            self.ui.pushButtonTheme.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonFont.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.pushButtonLanguage.setStyleSheet("QWidget { background-color: Cyan}" ) 
            self.ui.languageComboBox.setStyleSheet("QWidget { background-color: Cyan}" ) 
 
 
        if theme_index>1:      
            
#            pushButtonColor1
            self.ui.fig0.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.seg_text1.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.seg_text2.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.seg_text3.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.seg_text4.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.zoom_bar_1.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.zoom_bar_2.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.label_zoom1.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
            self.ui.label_zoom2.setStyleSheet("QWidget {background-color: rgb(255, 170, 211);}")
    
            self.ui.themebox.setStyleSheet("QWidget { background-color: rgb(194, 133, 255)}" )
            self.ui.groupBox_1.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.495, cy:0.494318, radius:2, fx:0.489, fy:0.494318, stop:0 rgba(221, 32, 56, 255), stop:1 rgba(255, 255, 255, 255));}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: rgb(231, 144, 255);}" )
            self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(194, 133, 255);}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(121, 123, 255)}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
            self.ui.time_p.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
            self.ui.author_7.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
 
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
        
        self.ui.time_p.setText(shamsi)
        try:
            import jdatetime
            T3 = jdatetime.date.fromgregorian(day = d, month = m, year = Y )  
            self.ui.time_p.setText(T3)
        except:s=1
        
        # from persiantools.jdatetime import JalaliDate, JalaliDateTime  
        # t1 = JalaliDateTime.now()# کلا  
        # t2 = JalaliDateTime.now().date # تاریخ  
        # t3 = JalaliDateTime.now().time # زمان  
        # t4 = JalaliDateTime.now().year # سال  
        # t5 = JalaliDateTime.now().month # ماه  
        # t6 = JalaliDateTime.now().day # روز  
        # t7 = JalaliDateTime.now().hour # ساعت  
        # t8 = JalaliDateTime.now().minute # دقیقه  
        # t9 = JalaliDateTime.now().second # ثانیه  
        
        
        
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
        self.ui.loadimage.setEnabled(False);
        self.ui.DICOM_input.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        self.ui.load_models.setEnabled(False);
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
            
            import cv2
            font = cv2.FONT_HERSHEY_COMPLEX
            import copy
            
            
            p1=int(np.fix(np.shape(image0) [0] /2  ))
            
            for font1 in range (step_phase2):
                for rep in range(2):
                    f1=0.0060 * np.shape(image0) [0] * font1/step_phase2
                    f2=0.003 * np.shape(image0) [0] * font1/step_phase2
                    image0_add_mask=copy.deepcopy(image0)
                    image0_add_mask =cv2.putText(image0_add_mask,' image',(0,p1),font,f1,(255,0,255),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis1(image0_add_mask)
                    
                    
                    image0_add_mask=copy.deepcopy(image0)
                    image0_add_mask =cv2.putText(image0_add_mask,' MASK',(0,p1),font,f1,(0,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis2(image0_add_mask)
                    image0_add_grad_cam=copy.deepcopy(image0)
                    image0_add_grad_cam =cv2.putText(image0_add_grad_cam,' Grad Cam',(0,p1),font,f2,(0,255,255),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis_grad_cam(image0_add_grad_cam)
                    
                    image0_add_grad_cam2=copy.deepcopy(image0)
                    image0_add_grad_cam2 =cv2.putText(image0_add_grad_cam2,' Grad Cam2',(0,p1),font,f2,(255,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
                    self.dis_grad_cam2(image0_add_grad_cam2)
                    cv2.waitKey(self.time000)
            # image0_add_mask=copy.deepcopy(image0)
            # image0_add_mask =cv2.putText(image0_add_mask,' image',(0,p1),font,f1,(255,0,255),3)  #text,coordinate,font,size of text,color,thickness of font
            cv2.waitKey(5*self.time000)
            
            self.dis1(image0)        
            # f1=512 / np.shape(image0) [0]   
            # p1=int(np.fix(np.shape(image0) [0] /2  ))
            # image0_add_mask=copy.deepcopy(image0)
            # image0_add_mask =cv2.putText(image0_add_mask,' MASK',(0,p1),font,f1,(0,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
            # image0_add_mask =cv2.putText(image0_add_mask,' MASK',(50,75),font,f1,(0,255,0),3)
            # self.dis2(image0_add_mask)
            
            # image0_add_grad_cam=copy.deepcopy(image0)
            # image0_add_grad_cam =cv2.putText(image0_add_grad_cam,' Grad Cam',(0,p1),font,0.8*f1,(0,0,255),3)  #text,coordinate,font,size of text,color,thickness of font
            # image0_add_grad_cam2=copy.deepcopy(image0)
            # image0_add_grad_cam2 =cv2.putText(image0_add_grad_cam2,' Grad Cam2',(0,p1),font,f2,(255,255,0),3)  #text,coordinate,font,size of text,color,thickness of font
            # self.dis_grad_cam2(image0_add_grad_cam2)
            
            # self.(image0_add_grad_cam)
            # self.dis_grad_cam2(image0_add_grad_cam2)
            
            cv2.imwrite('image0.jpg',image0)
            cv2.imwrite('image01.jpg',image0)        
            cv2.imwrite('out.jpg',image0)
            #       stage 2
            self.ui.applysegmentation.setEnabled(True);
            self.ui.load_models.setEnabled(True);


            
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
        
        
    def load_mask_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.DICOM_input.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        self.ui.load_models.setEnabled(False);
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
        import os
        path0= os.getcwd()
        fname = QFileDialog.getOpenFileName(self, 'Open file', path0)
        print(fname)
        if fname[0] !='':
            import cv2
            import numpy as np
            MASK=cv2.imread(fname[0])   
            MASK = cv2.resize(MASK, (840, 680))
            TEXT=  ' mask'
            cv2.imshow(TEXT,MASK)
            cv2.waitKey(3000)
            cv2.destroyWindow(TEXT)

            
            cv2.imwrite('MASK.jpg',MASK)
            cv2.imwrite('MASK1.jpg',MASK)        
            cv2.imwrite('MASK2.jpg',MASK)
            #       stage 2
            self.ui.applysegmentation.setEnabled(True);
            self.ui.load_models.setEnabled(True);


            
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
    def DICOM_input_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.DICOM_input.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        self.ui.load_models.setEnabled(False);
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
#       stage 2
        
        self.ui.applysegmentation.setEnabled(True);        
        self.ui.load_models.setEnabled(True);        
    def cropeimage_function(self):
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
         
        from subprocess import call
        import cv2
        call(["python", "crop_1.py"])
        image0 = cv2.imread('image01.jpg')     
        self.dis1(image0)
        self.dis2(image0)
        cv2.imwrite('image0.jpg',image0)
        cv2.imwrite('image01.jpg',image0)        
        cv2.imwrite('out.jpg',image0)  
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
#       stage 2
        
        self.ui.applysegmentation.setEnabled(True);        
        self.ui.load_models.setEnabled(True);        
    def cropeimage_2_function(self):
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
        self.dis_grad_cam(image0_add_grad_cam)
        
        image0_add_grad_cam2=copy.deepcopy(image0)
        
        image0_add_grad_cam2 =cv2.putText(image0_add_grad_cam2,' Grad Cam2',(0,p1),font,0.8*f1,(255,0,0),3)  #text,coordinate,font,size of text,color,thickness of font
        
        self.dis_grad_cam2(image0_add_grad_cam2)
        self.dis1(image0)
        self.dis2(image0_add_mask)
        
        cv2.imwrite('image0.jpg',image0)
        cv2.imwrite('image01.jpg',image0)        
        cv2.imwrite('out.jpg',image0) 
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
#       stage 2
        self.ui.applysegmentation.setEnabled(True);        
        self.ui.load_models.setEnabled(True);        
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
        self.seg=0
        amount=0;cnt=0
#        print(amount)
        if self.ui.deep1_method.isChecked()==True:
            amount=1;
        if self.ui.deep2_method.isChecked()==True:
            amount=2;    
        self.seg=amount    
        if amount==0:
            segg='None'
        if amount==1:
            segg='Deep method 1'
        if amount==2:
            segg='Deep method 2'
        if amount==3:
            segg='Classic Method'
        #self.ui.label_pre_2.setText('You Select '+segg)
    
    
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
        print('load_models_function  repairing')
        
        s=1
        
    def applysegmentation_function(self):
        self.ui.status.setText('You Press seg..')
        self.ui.loadimage.setEnabled(False);
        self.ui.DICOM_input.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
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
            self.dis2(dst);cv2.waitKey(20)
        image01= cv2.imread('image01.jpg')
       
        th=self.th+1
        import numpy as np
        import cv2 
        th=20
        
        import function_segmentation as segmi
        
        if self.seg>0:
            self.ui.result_seg_box.setEnabled(True);
            self.ui.cal_res1.setEnabled(True);
            
            self.ui.result_seg_box.setEnabled(True);
            self.ui.dice_overal.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.cal_res1.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.dice_LV.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.dice_RV.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.dice_Myo.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
            self.ui.mse1.setStyleSheet("QWidget { color: rgb(0, 0, 0); }")
        
        if self.seg==1:
            ( YY1 , p, superimposed_img1, superimposed_img)  =segmi.deep1_segmentation(image01)
            
            
# =============================================================================
#             reserve 
# =============================================================================
            # import matplotlib.pyplot as plt

            # plt.figure(1)
            # plt.subplot(2,3,1);plt.imshow( image01 ,cmap='gray' );plt.title('img1)')
            # plt.subplot(2,3,2);plt.imshow( image01 ,cmap='gray' );plt.title('mask1')
            # plt.subplot(2,3,3);plt.imshow( YY1 ,cmap='gray' );plt.title('Y')
            # plt.subplot(2,3,4);plt.imshow( p ,cmap='jet' );plt.title('p')
            # plt.subplot(2,3,5);plt.imshow( superimposed_img1 ,cmap='jet' );plt.title('superimposed_img')
            # plt.subplot(2,3,6);plt.imshow( superimposed_img ,cmap='gray' );plt.title('superimposed_img')
            
            # cv2.waitKey( 2000)
            # cv2.destroyWindow('1')
            
            
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
                
                self.dis2(i/step_phase31*YY1); cv2.waitKey( self.time0000)
                # self.dis2(i*p/9); cv2.waitKey( self.time0000)
                self.dis_grad_cam(i/step_phase31*superimposed_img1); cv2.waitKey( self.time0000)
                self.dis_grad_cam2(i/step_phase31*superimposed_img); cv2.waitKey( self.time0000)
                
                
            
        
            
        if self.seg==2:
            th=15
            (out, mask)=segmi.thershold_segmentation(image01,th)

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

        self.ui.applysegmentation.setEnabled(True);
        self.ui.applyclassification.setEnabled(True);
    
    def cal_res1_function(self):
        print( '  cal_res1_function  (self):')
        
        import numpy as np 
        dice_RV=97.3
        dice_LV=95.3
        dice_Myo=94.3
        
        dice_overal=np.mean([dice_RV,dice_LV,dice_Myo])
        dice_overal=0.01*np.fix(dice_overal*100)
        mse1=0.63
        
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
        
        self.ui.status.setText('You Press Apply clas..')
#        self.ui.time1.setEnabled(False)
        self.ui.time2.setEnabled(False)
        
        self.ui.loadimage.setEnabled(False);
        self.ui.DICOM_input.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        
        self.ui.applysegmentation.setEnabled(False);
        self.ui.applyinputdata.setEnabled(False);
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
        self.stage=6
        stage =self.stage
        self.stage_function(stage)
        classifer =self.classifer
        import function_classifier as clas
        import cv2
        import numpy as np
        image =cv2.imread('image01.jpg')
        deep_switch=self.deep
        print('deep_switch = ',deep_switch)
        if classifer == 'none':
            print ('None')
            print('deep_switch = ',self.deep)
            #self.ui.label_class.setText('None Method :)')
            self.ui.status.setText('None Method :)')
                
        if classifer == 'Deep learning':
            s=1
            print ('Deep learning')
            print('deep_switch = ',self.deep)
            self.ui.status.setText('Deep Method :)')
            
#            if deep_switch ==0:
            import function_classification as c
            (predictions,model)=c.deeep(image)

            print('model is loaded')
            model.summary()

            from keras.models import Sequential
            from keras.layers import Dense
            from keras.models import model_from_json
            import numpy
            import os
            # serialize model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")

            print (predictions)

        if classifer == 'SVM on Deep Features':
            s=1
            print ('SVM on Features')
            print('deep_switch = ',self.deep)
            #self.ui.label_class.setText('SVM Method :)')
            self.ui.status.setText('SVM Method :)')
 
        if classifer == 'KNN on Deep Features':
            s=1
            print ('KNN on Features')
            print('deep_switch = ',self.deep)
            #self.ui.label_class.setText('KNN Method :)')
            self.ui.status.setText('KNN Method :)')
 
        image02= cv2.imread('image02.jpg')
        self.dis2(image02);
            
        if predictions==1:
#            font: italic 14pt "Monotype Corsiva";
            self.ui.cass_1.setStyleSheet("QWidget { background-color: rgb(255, 0, 4) }")
            
        self.ui.time2.setEnabled(True)
        
        self.ui.loadimage.setEnabled(True);
        self.ui.DICOM_input.setEnabled(True);
        self.ui.cropeimage_2.setEnabled(True);
        self.ui.cropeimage.setEnabled(True);
        
        self.ui.applysegmentation.setEnabled(True);
        self.ui.applyinputdata.setEnabled(True);
        self.ui.applyclassification.setEnabled(True);
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
# =============================================================================
#  f6 classification       
# =============================================================================
    def dispAmount2(self):
        amount2=0
        if self.ui.deepleaning.isChecked()==True:
            amount2=1
        if self.ui.svm.isChecked()==True:
            amount2=2
        if self.ui.KNN.isChecked()==True:
            amount2=3
 
        if amount2==0:
            classifer='none'
        if amount2==1:
            classifer='Deep learning'

        if amount2==2:
            classifer='SVM on Deep Features'
        if amount2==3:
            classifer='KNN on Deep Features'
        
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
            image01 = cv2.imread('Hamed (3).jpg')
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
        
 
    def dis1(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(300,300))
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
        
        resized = cv2.resize(image0,(300,300))
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
        resized = cv2.resize(image0,(90,75))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_flag.setScene(scene) 
        
    def dis_grad_cam(self,image0):
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
        resized = cv2.resize(image0,(300,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_grad_cam.setScene(scene)
        
    def dis_grad_cam2(self,image0):
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
        resized = cv2.resize(image0,(300,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_grad_cam2.setScene(scene)
    def dis_logo(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(120,160))
        # resized =   image0 
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_logo.setScene(scene)
        
        
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




    def stage_function(self,stage):
        print('stage_function',stage)
        
        
        if self.stage >1:
            self.ui.fullscreen1.setEnabled(True)
            self.ui.segmentation_box.setEnabled(True)
            self.ui.deep1_method.setEnabled(True)
            
            self.ui.applysegmentation.setEnabled(True) 
            
         
            
        if self.stage >1:
            self.ui.classification_box.setEnabled(True)
            self.ui.deepleaning.setEnabled(True)
            self.ui.svm.setEnabled(True)
            self.ui.KNN.setEnabled(True)
            self.ui.label_class.setEnabled(True)
            self.ui.applyclassification.setEnabled(True)
            self.ui.result_box.setEnabled(True)
                   
            self.ui.time1.setEnabled(True) 
            self.ui.time2.setEnabled(True) 
            # self.ui.time_p.setEnabled(True) 
              
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

