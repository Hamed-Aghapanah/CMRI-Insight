import datetime
from time import gmtime, strftime
import sys
import cv2
from PyQt5 import QtWidgets, QtGui
# from PIL import Image, ImageQt
import numpy as np
from time import gmtime, strftime
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QApplication, QColorDialog,QFontDialog
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog
from G2 import *

import cv2
import numpy as np
src1=cv2.imread( 'images/Slide1.PNG')
image0 = cv2.resize(src1, (840, 680))

TEXT=115*' '+' CIAG Group'
cv2.imshow(TEXT,image0)
cv2.waitKey(5000)

cv2.destroyWindow(TEXT)

class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.datee=0
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionloadimage.triggered.connect(self.loadimage_function)
        self.ui.actionloadimage.triggered.connect(self.load_mask_function)
        self.ui.actioncamera.triggered.connect(self.camera_function)
        self.ui.actioncrop.triggered.connect(self.cropeimage_function)
        self.ui.actionsaveimage.triggered.connect(self.saveimage_function)
        # self.ui.actionshadowremoval.triggered.connect(self.applypreprocessing_function)
        # self.ui.actionhairremoval.triggered.connect(self.applypreprocessing_function)
        # self.ui.actionedgenhecement.triggered.connect(self.applypreprocessing_function)
        self.ui.actionEntropy.triggered.connect(self.applysegmentation_function)
        self.ui.actionThershold.triggered.connect(self.applysegmentation_function)
        # self.ui.actionage.triggered.connect(self.applyinputdata_function)
        # self.ui.actionsex.triggered.connect(self.applyinputdata_function)
        # self.ui.actionposition.triggered.connect(self.applyinputdata_function)
        self.ui.actionDeepLearning.triggered.connect(self.applyclassification_function)
        self.ui.actionSVM.triggered.connect(self.applyclassification_function)
        self.ui.actionKNN.triggered.connect(self.applyclassification_function)
        
#        self.ui.actionabout_us.triggered.connect(self.about_us_function)
        self.ui.actionHelp_Software.triggered.connect(self.Help_Software_function)
#        self.ui.actionvisit_Out_site.triggered.connect(self.Visit_Our_site_function)
#        self.ui.actionVisit_Our_site.triggered.connect(self.Visit_Our_site_function)
#        self.ui.actionVisit_software_page
        self.ui.actionVisit_software_page.triggered.connect(self.Visit_Our_site_function)
        self.ui.actionLicence.triggered.connect(self.Licence_function)
        
        
        self.ui.fig1.setEnabled(False);self.ui.fig2.setEnabled(False)
        self.ui.fig3.setEnabled(False);self.ui.fig4.setEnabled(False)
        self.show()
# =============================================================================
# section 0 : banner and initial images
# =============================================================================
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
        import cv2
        image0=cv2.imread('banner 2.jpg')
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
        self.dis_author(image_author)
        # self.dis_flag(image0)
        self.classifer='Deep learning'
        import time
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
        
        import calendar
        import time
        import numpy as np
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
        
        print('T1 = ',T1)
        print('T2 = ',T2)
        
        try:
            Y=T2[0:4];m=T2[5:7]; 
        except:
            Y=2023;   m=5;d=15
        try:    
            Y=np.int(Y)
        except:
            Y=2023;  m=5;d=15
        try:
            m=np.int(m)
        except:
            m=int(m)
            
        # try:
        #     d=np.int(d)
        # except:
        #     d=int(d)
        a=calendar.month(Y, m )
        self.ui.time1.setText(T1)
        self.ui.time2.setText(T2)
        self.ui.dateee.setText(a)
            
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
        self.ui.time2.clicked.connect(self.time2_function)
# =============================================================================
# section 1 : showing 
# =============================================================================
        self.ui.loadimage.clicked.connect(self.loadimage_function)
        self.ui.load_mask.clicked.connect(self.load_mask_function)
        self.ui.camera.clicked.connect(self.camera_function)
        self.ui.cropeimage.clicked.connect(self.cropeimage_function)
        self.ui.cropeimage_2.clicked.connect(self.cropeimage_2_function)

# =============================================================================
# section 2 : preprocessing         
# =============================================================================
        # # self.ui.hairremoval.stateChanged.connect(self.dispAmount)
        # # # self.ui.shadowremoval.stateChanged.connect(self.dispAmount)
        # # # self.ui.edgeenhacement.stateChanged.connect(self.dispAmount)
        # self.ui.manualpreprocessing.clicked.connect(self.manualpreprocessing_function)
        # self.ui.applypreprocessing.clicked.connect(self.applypreprocessing_function)
        # # # # self.ui.pre_bar.valueChanged.connect(self.pre_bar_function)
        # self.ui.zoom_bar_1.valueChanged.connect(self.zoom_bar_1_function)
        # self.ui.zoom_bar_2.valueChanged.connect(self.zoom_bar_2_function)
        # self.show()
        # 
# =============================================================================
# section 3 : Segmentation         
# =============================================================================
        self.ui.deep1_method.toggled.connect(self.dispAmount3)
        self.ui.deep2_method.toggled.connect(self.dispAmount3)
        self.ui.classic_method.toggled.connect(self.dispAmount3)
        self.ui.manualsegmentation.clicked.connect(self.manualsegmentation_function)
        self.ui.applysegmentation.clicked.connect(self.applysegmentation_function)
        # self.ui.segmentationbar.valueChanged.connect(self.segmentationbar_function)
# =============================================================================
# section 4 : Input data         
# =============================================================================
        # self.ui.applyinputdata.clicked.connect(self.applyinputdata_function)
    
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
            self.ui.input_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.classification_box.setStyleSheet("QWidget { background-color: %s }" % col.name())
            self.ui.result_box.setStyleSheet("QWidget { background-color: %s }" % col.name())


    def pushButtonfont_function(self):
        font, ok = QFontDialog.getFont()
        if ok:
#            font: 75 12pt "Times New Roman";
#            font: italic 14pt "Monotype Corsiva";
#            font: 8pt "Dark Horse";
#            self.ui.segmentation_box.setFont(font)
            self.ui.loadimage.setFont(font)  
#            self.ui.camera.setStyleSheet("QWidget { font: font: 75 12pt "Times New Roman" }" )
#            self.ui.cropeimage_2.setStyleSheet("QWidget { font: 8pt "Dark Horse" }")  
#            self.ui.cropeimage.setStyleSheet("QWidget { font: %s }" % font())  
#            # # self.ui.edgeenhacement.setStyleSheet("QWidget { font: %s }" % font())  
#            # self.ui.hairremoval.setStyleSheet("QWidget { font: %s }" % font())  
#            # # self.ui.shadowremoval.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.manualpreprocessing.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.applypreprocessing.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.original_image.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.result_image.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.nv_2.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.nv_2.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.nv_2.setStyleSheet("QWidget { font: %s }" % font())  
#            self.ui.nv_2.setStyleSheet("QWidget { font: %s }" % font())  
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
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Load Image')
            self.ui.camera.setText('Camera')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Crop')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Edge Enhancement')
            # self.ui.hairremoval.setText('Hair Removal')
            # # self.ui.shadowremoval.setText('Shadow Removal')
            # self.ui.manualpreprocessing.setText('Manual')
            # self.ui.applypreprocessing.setText('Apply')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.classic_method.setText('Threshold Method')
            self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Apply')                       
#            #self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('Asymmetric')
            ##self.ui.boundary.setText('Boundary')
            #self.ui.color.setText('Color')
            #self.ui.diameter.setText('Diameter')
            #self.ui.abcd1_5.setText('Sum  Score:')           
            #self.ui.abcd.setText('ABCD Result')           
#            self.ui.input_box.setText('input')
            ##self.ui.age_var.setText('Age')
            # self.ui.sex_var.setText('Sex')
            # self.ui.position_var.setText('Position')
            # self.ui.applyinputdata.setText('Apply')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Deep learning')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Apply')             
            #self.ui.abcd_4.setText('Result')             
            self.ui.original_image.setText('Original Image')             
            self.ui.result_image.setText('Result Image')             
            self.ui.saveimage.setText('Save Image')
            self.ui.fullscreen1.setText('Full Screen')
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
#            self.ui.groupBox_1.setText('Input Image')
            self.ui.loadimage.setText('Charger une image')
            self.ui.camera.setText('Caméra')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Surgir')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Amélioration des contours')
            # # self.ui.hairremoval.setText('Épilation')
            # # self.ui.shadowremoval.setText('Enlèvement des ombres')
            # self.ui.manualpreprocessing.setText('Manual')
            # self.ui.applypreprocessing.setText('Appliquer')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Méthode dentropie')
            self.ui.deep2_method.setText('Méthode dentropie')
            self.ui.classic_method.setText('Méthode de seuil')
            self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Appliquer')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('Asymétrique')
            #self.ui.boundary.setText('Frontière')
            #self.ui.color.setText('Couleur')
            #self.ui.diameter.setText('Diamètre')
            #self.ui.abcd1_5.setText('Somme Score:')           
            #self.ui.abcd.setText('ABCD Résultat')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Âge')
            #self.ui.sex_var.setText('Sexe')
            # self.ui.position_var.setText('Position')
            # self.ui.applyinputdata.setText('Appliquer')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Lapprentissage en profondeur')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Appliquer')             
            #self.ui.abcd_4.setText('Résultat')             
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
            self.ui.camera.setText('Camera')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Crop')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Edge Enhancement')
            # # self.ui.hairremoval.setText('Ontharing')
            # # self.ui.shadowremoval.setText('Schaduwverwijdering')
            # self.ui.manualpreprocessing.setText('Handleiding')
            # self.ui.applypreprocessing.setText('Van toepassing zijn')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.classic_method.setText('Threshold Method')
            self.ui.manualsegmentation.setText('Handleiding')
            self.ui.applysegmentation.setText('Van toepassing zijn')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('asymmetrisch')
            #self.ui.boundary.setText('Grens')
            ##self.ui.color.setText('Kleur')
            #self.ui.diameter.setText('Diameter')
            #self.ui.abcd1_5.setText('Somscore:')           
            #self.ui.abcd.setText('ABCD-resultaat')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Leeftijd')
            #self.ui.sex_var.setText('Seks')
            # self.ui.position_var.setText('Positie')
            # self.ui.applyinputdata.setText('Van toepassing zijn')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Diep leren')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Van toepassing zijn')             
            #self.ui.abcd_4.setText('Resultaat')             
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
            self.ui.camera.setText('कैमरा')
            self.ui.cropeimage_2.setText('ज़ूम')
            self.ui.cropeimage.setText('काटना')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('किनारा एनहांसमेंट')
            # # self.ui.hairremoval.setText('बाल हटाने वाला')
            # self.ui.shadowremoval.setText('छाया हटाना')
            # self.ui.manualpreprocessing.setText('गाइड')
            # self.ui.applypreprocessing.setText('लागू')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.classic_method.setText('Threshold Method')
            self.ui.manualsegmentation.setText('गाइड')
            self.ui.applysegmentation.setText('लागू')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('असममित')
            #self.ui.boundary.setText('सीमा')
            ##self.ui.color.setText('रंग')
            #self.ui.diameter.setText('व्यास')
            #self.ui.abcd1_5.setText('योग अंक:')           
            #self.ui.abcd.setText('ABCD का परिणाम')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('आयु')
            #self.ui.sex_var.setText('लिंग')
            # self.ui.position_var.setText('स्थान')
            # self.ui.applyinputdata.setText('लागू')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('ध्यान लगा के पढ़ना या सीखना')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('लागू')             
            #self.ui.abcd_4.setText('नतीजा')             
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
            self.ui.camera.setText('相机')
            self.ui.cropeimage_2.setText('放大')
            self.ui.cropeimage.setText('作物')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('边缘增强')
            # # self.ui.hairremoval.setText('除毛')
            # self.ui.shadowremoval.setText('去除阴影')
            # self.ui.manualpreprocessing.setText('手册')
            # self.ui.applypreprocessing.setText('应用')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.classic_method.setText('Threshold Method')
            self.ui.manualsegmentation.setText('手册')
            self.ui.applysegmentation.setText('应用')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('非对称')
            #self.ui.boundary.setText('边界')
            #self.ui.color.setText('颜色')
            #self.ui.diameter.setText('直径')
            #self.ui.abcd1_5.setText('总分:')           
            #self.ui.abcd.setText('ABCD结果')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('年龄')
            #self.ui.sex_var.setText('性别')
            #self.ui.position_var.setText('位置')
            # self.ui.applyinputdata.setText('应用')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('深度学习')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('应用')             
            #self.ui.abcd_4.setText('结果')             
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
            self.ui.camera.setText('Cámara')
            self.ui.cropeimage_2.setText('Enfocar')
            self.ui.cropeimage.setText('Cosecha')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Realce de bordes')
            # # self.ui.hairremoval.setText('Depilación')
            # self.ui.shadowremoval.setText('Shadow Removal')
            # self.ui.manualpreprocessing.setText('Manual')
            # self.ui.applypreprocessing.setText('Aplicar')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep method')
            self.ui.classic_method.setText('Threshold Method')
            self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Aplicar')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('Asimétrico')
            #self.ui.boundary.setText('Perímetro')
            #self.ui.color.setText('Color')
            ##self.ui.diameter.setText('Diámetro')
            #self.ui.abcd1_5.setText('Puntaje de suma:')           
            #self.ui.abcd.setText('ABCD Resultado')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Años')
            #self.ui.sex_var.setText('Sexo')
            #self.ui.position_var.setText('Posición')
            # self.ui.applyinputdata.setText('Aplicar')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Aprendizaje profundo')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Aplicar')             
            #self.ui.abcd_4.setText('Resultado')             
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
            self.ui.camera.setText('الة تصوير')
            self.ui.cropeimage_2.setText('تكبير')
            self.ui.cropeimage.setText('اقتصاص')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('تعزيز الحافة')
            # # self.ui.hairremoval.setText('مزيل شعر')
            # self.ui.shadowremoval.setText('إزالة الظل')
            # self.ui.manualpreprocessing.setText('كتيب')
            # self.ui.applypreprocessing.setText('تطبيق')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('طريقة الانتروبيا')
            self.ui.deep2_method.setText('طريقة الانتروبيا')
            self.ui.classic_method.setText('طريقة العتبة')
            self.ui.manualsegmentation.setText('كتيب')
            self.ui.applysegmentation.setText('تطبيق')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('لا متماثل')
            #self.ui.boundary.setText('حدود')
            #self.ui.color.setText('اللون')
            #self.ui.diameter.setText('قطر الدائرة')
            #self.ui.abcd1_5.setText('مجموع نقاط:')           
            #self.ui.abcd.setText('ABCD نتيجة')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('عمر')
            #self.ui.sex_var.setText('جنس')
            #self.ui.position_var.setText('موضع')
            # self.ui.applyinputdata.setText('تطبيق')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('تعلم عميق')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('تطبيق')             
            ##self.ui.abcd_4.setText('نتيجة ')             
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
            self.ui.camera.setText('Kamera')
            self.ui.cropeimage_2.setText('Zum')
            self.ui.cropeimage.setText('Potong')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Peningkatan Edge')
            # # self.ui.hairremoval.setText('Pembuangan Rambut')
            # self.ui.shadowremoval.setText('Pembuangan Bayangan')
            # self.ui.manualpreprocessing.setText('Manual')
            # self.ui.applypreprocessing.setText('Sapukan')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Kaedah Entropi')
            self.ui.deep2_method.setText('Kaedah Entropi')
            self.ui.classic_method.setText('Kaedah Ambang')
            self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Sapukan')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('Asimetri')
            #self.ui.boundary.setText('Sempadan')
            #self.ui.color.setText('Warna')
            #self.ui.diameter.setText('Diameter')
            ##self.ui.abcd1_5.setText('Markah Jumlah:')           
            ##self.ui.abcd.setText('ABCD Keputusan')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Umur')
            #self.ui.sex_var.setText('Seks')
            #self.ui.position_var.setText('Jawatan')
            # self.ui.applyinputdata.setText('Sapukan')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Pembelajaran yang mendalam')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Sapukan')             
            #self.ui.abcd_4.setText('Keputusan')             
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
            self.ui.camera.setText('камера')
            self.ui.cropeimage_2.setText('Увеличить')
            self.ui.cropeimage.setText('урожай')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Улучшение края')
            # # self.ui.hairremoval.setText('Удаление волос')
            # self.ui.shadowremoval.setText('Удаление Тени')
            # self.ui.manualpreprocessing.setText('Manual')
            # self.ui.applypreprocessing.setText('Применять')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Метод энтропии')
            self.ui.deep2_method.setText('Метод энтропии')
            self.ui.classic_method.setText('Пороговый метод')
            self.ui.manualsegmentation.setText('Manual')
            self.ui.applysegmentation.setText('Применять')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('асимметричный')
            #self.ui.boundary.setText('граничный')
            #self.ui.color.setText('цвет')
            #self.ui.diameter.setText('Диаметр')
            #self.ui.abcd1_5.setText('Сумма очков:')           
            #self.ui.abcd.setText('ABCD Результат')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Возраст')
            #self.ui.sex_var.setText('секс')
            #self.ui.position_var.setText('Позиция')
            # self.ui.applyinputdata.setText('Применять')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Глубокое обучение')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Применять')             
            #self.ui.abcd_4.setText('Результат')             
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
            self.ui.camera.setText('telecamera')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Ritaglia')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Miglioramento dei bordi')
            # # self.ui.hairremoval.setText('Rimozione peli')
            # self.ui.shadowremoval.setText('Rimozione dell ombra')
            # self.ui.manualpreprocessing.setText('Manuale')
            # self.ui.applypreprocessing.setText('Applicare')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Metodo di entropia')
            self.ui.deep2_method.setText('Metodo di entropia')
            self.ui.classic_method.setText('Metodo di soglia')
            self.ui.manualsegmentation.setText('Manuale')
            self.ui.applysegmentation.setText('Applicare')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('Asimmetrico')
            #self.ui.boundary.setText('Confine')
            #self.ui.color.setText('Colore')
            #self.ui.diameter.setText('Diametro')
            #self.ui.abcd1_5.setText('Somma punteggio:')           
            #self.ui.abcd.setText('ABCD Risultato')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Età')
            #self.ui.sex_var.setText('Sesso')
            #self.ui.position_var.setText('Posizione')
            # self.ui.applyinputdata.setText('Applicare')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Apprendimento profondo')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Applicare')             
            #self.ui.abcd_4.setText('Risultato')             
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
            self.ui.camera.setText('دوربین')
            self.ui.cropeimage_2.setText('بزرگ نمایی')
            self.ui.cropeimage.setText('برش')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('بهبود لبه')
            # # self.ui.hairremoval.setText('حذف مو')
            # self.ui.shadowremoval.setText('حذف سایه')
            # self.ui.manualpreprocessing.setText('دستی')
            # self.ui.applypreprocessing.setText('اعمال')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('روش عمیق1')
            self.ui.deep2_method.setText('روش عمیق2')
            self.ui.classic_method.setText('روش آستانه گذاری')
            self.ui.manualsegmentation.setText('دستی')
            self.ui.applysegmentation.setText('اعمال')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('عدم تقارن')
            #self.ui.boundary.setText('لبه')
            #self.ui.color.setText('رنگ')
            #self.ui.diameter.setText('قطر')
            #self.ui.abcd1_5.setText('جمع:')           
            #self.ui.abcd.setText('ABCD نتیجه')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('سن')
            #self.ui.sex_var.setText('جنس')
            #self.ui.position_var.setText('رنگ')
            # self.ui.applyinputdata.setText('اعمال')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('یادگیری عمیق')
            self.ui.svm.setText('ماشین بردار پشتیبان')
            self.ui.KNN.setText('نزدیکترین همسایگی')
            self.ui.applyclassification.setText('اعمال')             
            #self.ui.abcd_4.setText('نتیجه')             
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
            self.ui.camera.setText('Camera')
            self.ui.cropeimage_2.setText('Zoom')
            self.ui.cropeimage.setText('Crop')
#            # self.ui.preprocessing_box.setText('preprocessing')
            # self.ui.edgeenhacement.setText('Edge Enhancement')
            # # self.ui.hairremoval.setText('Ontharing')
            # self.ui.shadowremoval.setText('Schaduwverwijdering')
            # self.ui.manualpreprocessing.setText('Handleiding')
            # self.ui.applypreprocessing.setText('Van toepassing zijn')          
#            self.ui.segmentation_box.setText('segmentation')
            self.ui.deep1_method.setText('Deep method')
            self.ui.deep2_method.setText('Deep2 method')
            self.ui.classic_method.setText('Threshold Method')
            self.ui.manualsegmentation.setText('Handleiding')
            self.ui.applysegmentation.setText('Van toepassing zijn')                       
#            self.ui.clinical_box.setText('clinical')
           # self.ui.asymetric.setText('asymmetrisch')
            #self.ui.boundary.setText('Grens')
            #self.ui.color.setText('Kleur')
            #self.ui.diameter.setText('Diameter')
            #self.ui.abcd1_5.setText('Somscore:')           
            #self.ui.abcd.setText('ABCD-resultaat')           
#            self.ui.input_box.setText('input')
            #self.ui.age_var.setText('Leeftijd')
            #self.ui.sex_var.setText('Seks')
            #self.ui.position_var.setText('Positie')
            # self.ui.applyinputdata.setText('Van toepassing zijn')
#            self.ui.classification_box.setText('classification')
            self.ui.deepleaning.setText('Diep leren')
            self.ui.svm.setText('SVM on Deep Features')
            self.ui.KNN.setText('KNN on Deep Features')
            self.ui.applyclassification.setText('Van toepassing zijn')             
            #self.ui.abcd_4.setText('Resultaat')             
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
            self.ui.input_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color:rgb(87, 112, 255) }" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(87, 112, 255)}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: black}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: black}" ) 
            self.ui.author_7.setStyleSheet("QWidget { background-color: rgb(87, 112, 255);}" ) 
         
            self.ui.loadimage.setStyleSheet("QWidget { background-color: Green }" )             
            self.ui.camera.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.cropeimage.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.cropeimage_2.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.saveimage.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Green}" )             
           
            self.ui.applysegmentation.setStyleSheet("QWidget { background-color: Green}" )             
            # self.ui.manualpreprocessing.setStyleSheet("QWidget { background-color: Green}" )             
            # self.ui.applypreprocessing.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.manualsegmentation.setStyleSheet("QWidget { background-color: Green}" )             
            # self.ui.applyinputdata.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.applyclassification.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.mel_2.setStyleSheet("QWidget { background-color: Green}" )             
            #self.ui.scc_2.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.bk_2.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.bcc_2.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.df_2.setStyleSheet("QWidget { background-color: Green}" )             
            self.ui.nv_2.setStyleSheet("QWidget { background-color: Green}" )             
            #self.ui.ak_2.setStyleSheet("QWidget { background-color: Green}" ) 
            # self.ui.vasc_2.setStyleSheet("QWidget { background-color: Green}" ) 
            
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
            # self.ui.preprocessing_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.clinical_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.input_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: bold gray}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: bold gray}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color: rgb(187, 212, 255)}" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(187, 212, 255)}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: white}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: white}" ) 
            self.ui.author_7.setStyleSheet("QWidget { background-color: bold gray}" ) 

            self.ui.loadimage.setStyleSheet("QWidget { background-color: Cyan }" )             
            self.ui.camera.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.cropeimage.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.cropeimage_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.saveimage.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: Cyan}" )             
           
            self.ui.applysegmentation.setStyleSheet("QWidget { background-color: Cyan}" )             
            # self.ui.manualpreprocessing.setStyleSheet("QWidget { background-color: Cyan}" )             
            # self.ui.applypreprocessing.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.manualsegmentation.setStyleSheet("QWidget { background-color: Cyan}" )             
            # self.ui.applyinputdata.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.applyclassification.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.mel_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            #self.ui.scc_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.bk_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.bcc_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.df_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            self.ui.nv_2.setStyleSheet("QWidget { background-color: Cyan}" )             
            #self.ui.ak_2.setStyleSheet("QWidget { background-color: Cyan}" ) 
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
            # self.ui.preprocessing_box.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.489, cy:0.5, radius:2, fx:0.489, fy:0.494318, stop:0 rgba(255, 119, 224, 255), stop:1 rgba(255, 255, 255, 255));}" )
            self.ui.segmentation_box.setStyleSheet("QWidget { background-color: rgb(231, 144, 255);}" )
            self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(194, 133, 255);}" )
            self.ui.input_box.setStyleSheet("QWidget { background-color: rgb(152, 111, 255);}" )
            self.ui.classification_box.setStyleSheet("QWidget { background-color: rgb(121, 123, 255)}" )
            self.ui.result_box.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" )             
            self.ui.original_image.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" )             
            self.ui.result_image.setStyleSheet("QWidget { background-color: rgb(123, 176, 255);}" )             
            self.ui.time2.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" )             
            self.ui.time1.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
            self.ui.author_7.setStyleSheet("QWidget { background-color: rgb(85, 201, 255);}" ) 
 
            self.ui.loadimage.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255)); }" )             
            self.ui.camera.setStyleSheet("QWidget {  background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255));}")             
            self.ui.cropeimage.setStyleSheet("QWidget {  background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255))};" )             
            self.ui.cropeimage_2.setStyleSheet("QWidget {  background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255))};" )             
            self.ui.saveimage.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:2, fx:0.5, fy:0.5, stop:0 rgba(240, 101, 142, 237), stop:1 rgba(255, 255, 255, 255));}" )             
            self.ui.pushButtonFont.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" )             
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 

            # self.ui.applypreprocessing.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.489, cy:0.5, radius:2, fx:0.489, fy:0.494318, stop:0 rgba(255, 169, 224, 255), stop:1 rgba(255, 255, 255, 255))}" )             
            # self.ui.manualpreprocessing.setStyleSheet("QWidget { background-color: qradialgradient(spread:pad, cx:0.489, cy:0.5, radius:2, fx:0.489, fy:0.494318, stop:0 rgba(255, 169, 224, 255), stop:1 rgba(255, 255, 255, 255));}" )             
            self.ui.applysegmentation.setStyleSheet("QWidget { background-color: rgb(231, 194, 255);}" )             
            self.ui.manualsegmentation.setStyleSheet("QWidget { background-color: rgb(231, 194, 255);}" )             
            # self.ui.applyinputdata.setStyleSheet("QWidget { background-color: rgb(202, 161, 255);}" )             
            self.ui.applyclassification.setStyleSheet("QWidget {background-color: rgb(171, 173, 255);}" )             
            self.ui.mel_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" )             
            #self.ui.scc_2.setStyleSheet("QWidget {background-color: rgb(158, 221, 255);}" )             
            self.ui.bk_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" )             
            self.ui.bcc_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" )             
            self.ui.df_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" )             
            self.ui.nv_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" )             
            #self.ui.ak_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" ) 
            #self.ui.vasc_2.setStyleSheet("QWidget { background-color: rgb(158, 221, 255);}" ) 
#            pushButtonFont
            self.ui.pushButtonTheme.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonColor1.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonColor2.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
            self.ui.pushButtonLanguage.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
#            self.ui.fontComboBox.setStyleSheet("QWidget { background-color: rgb(85, 170, 255);}" ) 
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
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
        print(T1)
        self.ui.time1.setText(T1)
        self.ui.time2.setText(T2)
    def time2_function(self):
        import calendar 
        import time
        import numpy as np
        T1=strftime("%H:%M:%S", time.localtime())
        T2=strftime("%Y-%m-%d ", time.localtime())
#        print(T2)
        Y=T2[0:4]
        m=T2[5:7]
        print (T2)
        print (Y)
        print (m)
        Y=np.int(Y)
        
        try:
            m=np.int(m)
        except:
            m=int(m)
            
        a=calendar.month(Y, m)
        print(a)
        self.ui.time1.setText(T1)
        self.ui.time2.setText(T2)
        if self.datee==1:
            self.datee=0
#            print ('insert calendar')
#            print (T2)
            self.ui.dateee.setText(a)
        else:
            self.datee=1
#            print ('clear calendar')
#            print (T2)
            self.ui.dateee.setText('Calendar')

# =============================================================================
#   f1
# ============================================================================= 
    def loadimage_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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
#            fname = QFileDialog.getOpenFileName(self, 'Open file', 'please select an image')
#            print('نشد')      
            import cv2
            import numpy as np
            image0=cv2.imread(fname[0])          
            self.image0 = image0
            self.dis1(image0)
            self.stage=3
            stage =self.stage
            self.stage_function(stage)
            self.dis1(image0)
            self.dis2(image0)
            cv2.imwrite('image0.jpg',image0)
            cv2.imwrite('image01.jpg',image0)        
            cv2.imwrite('out.jpg',image0)
            #       stage 2
            # self.ui.applypreprocessing.setEnabled(True);
            # self.ui.manualpreprocessing.setEnabled(True);
            self.ui.manualsegmentation.setEnabled(True);
            self.ui.applysegmentation.setEnabled(True);


            
#        stage 1 
        self.ui.loadimage.setEnabled(True);
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);
        
        
    def load_mask_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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
#            fname = QFileDialog.getOpenFileName(self, 'Open file', 'please select an image')
#            print('نشد')      
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
            # self.ui.applypreprocessing.setEnabled(True);
            # self.ui.manualpreprocessing.setEnabled(True);
            self.ui.manualsegmentation.setEnabled(True);
            self.ui.applysegmentation.setEnabled(True);


            
#        stage 1 
        self.ui.loadimage.setEnabled(True);
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);
    def camera_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);
#       stage 2
        # self.ui.applypreprocessing.setEnabled(True);
        # self.ui.manualpreprocessing.setEnabled(True);
        self.ui.manualsegmentation.setEnabled(True);
        self.ui.applysegmentation.setEnabled(True);        
    def cropeimage_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);        
#       stage 2
        # self.ui.applypreprocessing.setEnabled(True);
        # self.ui.manualpreprocessing.setEnabled(True);
        self.ui.manualsegmentation.setEnabled(True);
        self.ui.applysegmentation.setEnabled(True);        
    def cropeimage_2_function(self):
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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

        self.dis1(image0)
        self.dis2(image0)
        cv2.imwrite('image0.jpg',image0)
        cv2.imwrite('image01.jpg',image0)        
        cv2.imwrite('out.jpg',image0) 
#        stage 1 
        self.ui.loadimage.setEnabled(True);
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);        
#       stage 2
        # self.ui.applypreprocessing.setEnabled(True);
        # self.ui.manualpreprocessing.setEnabled(True);
        self.ui.manualsegmentation.setEnabled(True);
        self.ui.applysegmentation.setEnabled(True);        
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
#        print(amount)
        # if # self.ui.hairremoval.isChecked()==True:
        #     amount=amount+1;cnt=cnt+1
        # if # # self.ui.shadowremoval.isChecked()==True:
        #     amount=amount+10;cnt=cnt+1
        # if # self.ui.edgeenhacement.isChecked()==True:
        #     amount=amount+100;cnt=cnt+1
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
        ##self.ui.label_pre.setText('')
        ##self.ui.label_pre_2.setText(' ') 
        #self.ui.label_input.setText(' ')        
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
        if self.ui.classic_method.isChecked()==True:
            amount=2; 
        self.seg=amount    
        if amount==0:
            segg='None'
        if amount==1:
            segg='Deep method 1'
        if amount==2:
            segg='Deep method 2'
        if amount==3:
            segg='ژlassic Method'
        #self.ui.label_pre_2.setText('You Select '+segg)
    
    
    def manualsegmentation_function(self):
        #self.ui.label_pre.setText('')
        #self.ui.label_pre_2.setText(' ') 
        #self.ui.label_input.setText(' ')        
        #self.ui.label_class.setText(' ')
        #self.ui.label_pre_2.setText('You Press Manual seg..')
        self.ui.status.setText('You Press Manual seg..')
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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
        self.ui.pushButtonColor1.setEnabled(False);
        self.ui.pushButtonColor2.setEnabled(False);
        self.ui.pushButtonLanguage.setEnabled(False);
        self.ui.pushButtonFont.setEnabled(False); 
        self.ui.languageComboBox.setEnabled(False); 
        self.ui.ThemeComboBox.setEnabled(False);
        self.ui.input_box.setEnabled(False);
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
        import function_segmentation as seg
        out=seg.thershold_segmentation(image01,th)
        image02=out
        cv2.imwrite('image02.jpg',image02)
        self.stage=4
        stage =self.stage
        self.stage_function(stage)
        self.out=out
        self.out1=out1
        self.image2 = image2
        import abcd_Features as fea
        (features, img, cnt1)  = fea.abcd_Features(image1)
        a=features[0]
        b=features[1]
        c=features[2]
        d=features[3]
        
        import numpy as np
        abcd=a+b+c+d
        abcd=(np.fix(10*abcd))
        abcd=0.1*np.fix(abcd)
#        print('abcd',abcd)
       ## self.ui.asymetric1.setText(str(a))
        self.ui.boundry1.setText(str(b))
        #self.ui.color1.setText(str(c))
        #self.ui.diameter1.setText(str(d))
        #self.ui.abcd1.setText(str(abcd))
        self.ui.input_box.setEnabled(False);
        self.ui.age1.setEnabled(False);
        #self.ui.sex1.setEnabled(False);
        #self.ui.pos1.setEnabled(False);
        # if abcd>10:
        #     self.ui.abcd.setText('Melanoma')
        #     for i in range(10):
        #         #self.ui.clinical_box.setStyleSheet("QWidget { background-color: red }")
        #         cv2.waitKey(700)
        #         #self.ui.clinical_box.setStyleSheet("QWidget { background-color: yellow }")
        #         cv2.waitKey(700)
        # else:
        #     self.ui.abcd.setText('Ordinary :)')
        #     for i in range(10):
        #         #self.ui.clinical_box.setStyleSheet("QWidget { background-color: green }")
        #         cv2.waitKey(700)
        #         #self.ui.clinical_box.setStyleSheet("QWidget { background-color: blue }")
        #         cv2.waitKey(700)


        self.stage=5
        stage =self.stage
        self.stage_function(stage)
        
#        stage 1 
        self.ui.loadimage.setEnabled(True);
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);
#        stage 2
        # self.ui.applypreprocessing.setEnabled(True);
        # self.ui.manualpreprocessing.setEnabled(True);
        self.ui.manualsegmentation.setEnabled(True);
        self.ui.applysegmentation.setEnabled(True);        
#       stage 3 
        # self.ui.applyinputdata.setEnabled(True);
        self.ui.applyclassification.setEnabled(True);
        self.ui.input_box.setEnabled(True);
        # self.ui.age1.setEnabled(True);
        # self.ui.sex1.setEnabled(True);
        # self.ui.pos1.setEnabled(True);
    def applysegmentation_function(self):
        #self.ui.label_pre.setText('')
        #self.ui.label_pre_2.setText(' ') 
        #self.ui.label_input.setText(' ')        
        #self.ui.label_class.setText(' ')
        #self.ui.label_pre_2.setText('You Press manual seg..')
        self.ui.status.setText('You Press manual seg..')
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
        self.ui.applysegmentation.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
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
        self.ui.pushButtonColor1.setEnabled(False);
        self.ui.pushButtonColor2.setEnabled(False);
        self.ui.pushButtonLanguage.setEnabled(False);
        self.ui.pushButtonFont.setEnabled(False); 
        self.ui.languageComboBox.setEnabled(False); 
        self.ui.ThemeComboBox.setEnabled(False);
        # self.ui.applyinputdata.setEnabled(False);
        # self.ui.input_box.setEnabled(False);
        # self.ui.age1.setEnabled(False);
        # self.ui.sex1.setEnabled(False);
        # self.ui.pos1.setEnabled(False);
        
#        import cv2
#        cv2.waitKey(5000)
#        print('falsed  all?')
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
        
        import function_segmentation as seg
        if self.seg==1:
            (out, mask)=seg.deep1_segmentation(image01)
        if self.seg==2:
            th=15
            (out, mask)=seg.thershold_segmentation(image01,th)

        import cv2 
        import abcd_Features as fea
        image01 = cv2.imread('image01.jpg')
#        cv2.imshow('image1',image1)
        features=[]
        (features, img, cnt1)= fea.abcd_Features(image01)
        try:
            print(3*' fea.abcd_Feature')
            (features, img, cnt1)= fea.abcd_Features(image01)
            print(3*' sss.abcd_Feature')
            cv2.waitKey(2000)
            for i in range(60):
                if (i/7)==0:
                    cv2.drawContours(img,[cnt1],-1,(0,255,255),1),
                if (i/7)==1:
                    cv2.drawContours(img,[cnt1],-1,(255,0,255),2),
                if (i/7)==2:
                    cv2.drawContours(img,[cnt1],-1,(255,255,0),3),
                if (i/7)==3:
                    cv2.drawContours(img,[cnt1],-1,(255,0,0),4),
                if (i/7)==4:
                    cv2.drawContours(img,[cnt1],-1,(0,0,255),5),
                if (i/7)==5:
                    cv2.drawContours(img,[cnt1],-1,(0,255,0),0),
                if (i/7)==6:
                    cv2.drawContours(img,[cnt1],-1,(255,255,255),0),

                    self.dis2(img)

                cv2.waitKey(20)
        except:
            import random
            features=[random.randint(1, 99) /100 ,
                      random.randint(3, 99) /100,
                      random.randint(1, 99) /100,
                      random.randint(2, 99) /100]
            img = image01
            cnt1=4
        
        image02=img
#        cv2.destroyWindow('image1')
        cv2.imwrite('out.jpg',image02)
        cv2.imwrite('image02.jpg',image02)
#        cv2.imwrite('out1.jpg',out1)
        self.stage=4
        stage =self.stage
        self.stage_function(stage)
        a=features[0]
        b=features[1]
        c=features[2]
        d=features[3]
        import numpy as np
        abcd=a+b+c+d
        abcd=(np.fix(10*abcd))
        abcd=0.1*np.uint16(abcd)
#        print('abcd',abcd)
        # self.ui.applyinputdata.setEnabled(False);
        # self.ui.input_box.setEnabled(False);
        # self.ui.age1.setEnabled(False);
        # self.ui.sex1.setEnabled(False);
        # self.ui.pos1.setEnabled(False);
        if abcd>5:
            #self.ui.abcd.setText('Melanoma')
            for i in range(5):
                #self.ui.clinical_box.setStyleSheet("QWidget { background-color:  rgb(255, 222, 32);  }")
                cv2.waitKey(100)
                #self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(255, 6, 10) }")
                cv2.waitKey(300)
                #self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(194, 133, 255) }")
                cv2.waitKey(100)
        else:
            #self.ui.abcd.setText('Normal')
            for i in range(5):
                #self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(69, 215, 255)}")
                cv2.waitKey(100)
                #self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(52, 255, 99) }")
                cv2.waitKey(300)
                #self.ui.clinical_box.setStyleSheet("QWidget { background-color: rgb(194, 133, 255) }")
                cv2.waitKey(100)
                 
        
#        cv2.waitKey(5000)
#        print('falsed  all2?')
        print('saedddddd')
              
       ## self.ui.asymetric1.setText(str(a))
        self.ui.boundry1.setText(str(b))
        #self.ui.color1.setText(str(c))
        #self.ui.diameter1.setText(str(d))
        #self.ui.abcd1.setText(str(abcd))        
        self.stage=5
        stage =self.stage
        self.stage_function(stage)        
#        stage 1 
        self.ui.loadimage.setEnabled(True);
        self.ui.camera.setEnabled(True);
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
        self.ui.pushButtonColor1.setEnabled(True);
        self.ui.pushButtonColor2.setEnabled(True);
        self.ui.pushButtonLanguage.setEnabled(True);
        self.ui.pushButtonFont.setEnabled(True); 
        self.ui.languageComboBox.setEnabled(True); 
        self.ui.ThemeComboBox.setEnabled(True);        
#        stage 2
        # self.ui.applypreprocessing.setEnabled(True);
        # self.ui.manualpreprocessing.setEnabled(True);
        self.ui.manualsegmentation.setEnabled(True);
        self.ui.applysegmentation.setEnabled(True);
        # self.ui.applyinputdata.setEnabled(True);
        # self.ui.input_box.setEnabled(True);
        # self.ui.age1.setEnabled(True);
        # self.ui.sex1.setEnabled(True);
        # self.ui.pos1.setEnabled(True);        
#       stage 3 
#        self.ui.applyinputdata.setEnabled(True);
        self.ui.applyclassification.setEnabled(True);
    def segmentationbar_function(self, value):    
        #self.ui.label_pre.setText('')
        #self.ui.label_pre_2.setText(' ') 
        #self.ui.label_input.setText(' ')        
        #self.ui.label_class.setText(' ')
         
        import numpy as np
        #self.ui.label_pre_2.setText("Level : "+str(np.fix((2.55/0.99)* value)))
        self.th=1+np.fix((2.55/0.99)* value)
        
 
# =============================================================================
#  f5      
# =============================================================================
    def applyclassification_function(self):
        
        #self.ui.label_pre.setText('')
        #self.ui.label_pre_2.setText(' ') 
        #self.ui.label_input.setText(' ')        
        #self.ui.label_class.setText(' ')
        #self.ui.label_class.setText('You Press Apply clas..')
        self.ui.status.setText('You Press Apply clas..')
#        self.ui.time1.setEnabled(False)
        self.ui.time2.setEnabled(False)
        self.ui.mel_2.setEnabled(False)
        self.ui.bk_2.setEnabled(False)
        self.ui.nv_2.setEnabled(False)
        self.ui.df_2.setEnabled(False)
        self.ui.bcc_2.setEnabled(False)
        #self.ui.vasc_2.setEnabled(False)
        #self.ui.ak_2.setEnabled(False)
        ##self.ui.scc_2.setEnabled(False)
        self.ui.loadimage.setEnabled(False);
        self.ui.camera.setEnabled(False);
        self.ui.cropeimage_2.setEnabled(False);
        self.ui.cropeimage.setEnabled(False);
        # self.ui.applypreprocessing.setEnabled(False);
        # self.ui.manualpreprocessing.setEnabled(False);
        self.ui.manualsegmentation.setEnabled(False);
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
#        image01= cv2.imread('image01.jpg')
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
            #self.ui.label_class.setText('Deep Method :)')
            self.ui.status.setText('Deep Method :)')
            
#            if deep_switch ==0:
            import function_classification as c
            (predictions,model)=c.deeep(image)
#                self.model5=model5
#                model5=c.Model
#                deeep(image)
            print('model is loaded')
            model.summary()
#                self.model=model5
            # serialize model to JSON
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
#            self.deep=self.deep+1000               
#            else:
#                model5=self.model5
#                X_test = []
#                Target = []
#                for index in range(9):
#                    for i in range(3):
#                        img =image
#                        img_resized = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
#                        img_resized = img_resized / 255
#                        X_test.append(img_resized)
#                        Target.append(index)
#                X_test = np.array(X_test)
#                json_file = open('model.json', 'r')
#                loaded_model_json = json_file.read()
#                json_file.close()
#                loaded_model = model_from_json(loaded_model_json)
#                # load weights into new model
#                loaded_model.load_weights("model.h5")
#                print("Loaded model from disk")
# 
## evaluate loaded model on test data
#                loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
#                preds5=loaded_model.predict(X_test  , verbose=1)
#                decoded_output5 = np.argmax(preds5,axis=1)
#                predictions =  decoded_output5[1]+1
            
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
                
        self.ui.mel_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        self.ui.nv_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        self.ui.bcc_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        #self.ui.ak_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        self.ui.bk_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        self.ui.df_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        #self.ui.vasc_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
        ##self.ui.scc_2.setStyleSheet("QWidget { background-color: rgb(0, 0, 0 }")
     
        if predictions==1:
#            font: italic 14pt "Monotype Corsiva";
            self.ui.mel_2.setStyleSheet("QWidget { background-color: rgb(255, 0, 4) }")
            #self.ui.result_lable.setText('Melanoma')
         
            #self.ui.result_lable.setText('Dermatofibroma')
        # if predictions==7:
        #     self.ui.vasc_2.setStyleSheet("QWidget { background-color: rgb(255, 0, 4) }")
        #     #self.ui.result_lable.setText('Vascular lesion')
        # if predictions==8:
        #     #self.ui.scc_2.setStyleSheet("QWidget { background-color: rgb(255, 0, 4) }")
        #     #self.ui.result_lable.setText('Squamous cell carcinoma')
        # if predictions>8:
        #     self.ui.mel_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     self.ui.nv_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     self.ui.bcc_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     #self.ui.ak_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     self.ui.bk_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     self.ui.df_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     self.ui.vasc_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     #self.ui.scc_2.setStyleSheet("QWidget { background-color: rgb(255, 255, 255) }")
        #     ##self.ui.result_lable.setText('None of Them')
        self.ui.time2.setEnabled(True)
        self.ui.mel_2.setEnabled(True)
        self.ui.bk_2.setEnabled(True)
        self.ui.nv_2.setEnabled(True)
        self.ui.df_2.setEnabled(True)
        self.ui.bcc_2.setEnabled(True)
        #self.ui.vasc_2.setEnabled(True)
        #self.ui.ak_2.setEnabled(True)
        #self.ui.scc_2.setEnabled(True)
        self.ui.loadimage.setEnabled(True);
        self.ui.camera.setEnabled(True);
        self.ui.cropeimage_2.setEnabled(True);
        self.ui.cropeimage.setEnabled(True);
        # self.ui.applypreprocessing.setEnabled(True);
        # self.ui.manualpreprocessing.setEnabled(True);
        self.ui.manualsegmentation.setEnabled(True);
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
        #self.ui.label_pre.setText('')
        #self.ui.label_pre_2.setText(' ') 
        #self.ui.label_input.setText(' ')        
        #self.ui.label_class.setText(' ')
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
        resized = cv2.resize(image0,(450,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig1.setScene(scene)
        
        
        
    def dis2(self,image0):
        import cv2
        resized = cv2.resize(image0,(450,300))
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
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(450,300))
        ResultImage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)      
        height, width, bytesPerComponent = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(QImg)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.fig_grad_cam.setScene(scene)
    def dis_logo(self,image0):
        import cv2
        import numpy as np
        x0 = np.size(image0,0);
        x1 = np.size(image0,1)
        resized = cv2.resize(image0,(140,180))
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
            self.ui.deep2_method.setEnabled(True)
            self.ui.classic_method.setEnabled(True)
            # self.ui.segmentationbar.setEnabled(True)
            self.ui.manualsegmentation.setEnabled(True)
            self.ui.applysegmentation.setEnabled(True) 
            # self.ui.seg2.setEnabled(True) 
            # self.ui.seg1.setEnabled(True) 
            # self.ui.line_seg.setEnabled(True) 
            # self.ui.fullscreen2.setEnabled(True)
            #self.ui.label_pre_2.setEnabled(True)
#            col=[ 0 , 0 , 0 ]
#            self.ui.fullscreen2.setStyleSheet("QWidget { background-color: %s }" % col.name())      
            
            #self.ui.clinical_box.setEnabled(True)
           # self.ui.asymetric.setEnabled(True)
            #self.ui.boundary.setEnabled(True)
            #self.ui.color.setEnabled(True)
            #self.ui.diameter.setEnabled(True)
           # self.ui.asymetric1.setEnabled(True)
            # self.ui.boundry1.setEnabled(True)
            #self.ui.color1.setEnabled(True)
            #self.ui.diameter1.setEnabled(True)
            #self.ui.abcd.setEnabled(True)        
            #self.ui.abcd1.setEnabled(True) 
            #self.ui.abcd1_5.setEnabled(True) 
            # self.ui.line_abcd.setEnabled(True) 
            #self.ui.label_input.setEnabled(True) 
         
            
        if self.stage >1:
            self.ui.classification_box.setEnabled(True)
            self.ui.deepleaning.setEnabled(True)
            self.ui.svm.setEnabled(True)
            self.ui.KNN.setEnabled(True)
            self.ui.label_class.setEnabled(True)
            # self.ui.line_class.setEnabled(True)
            self.ui.applyclassification.setEnabled(True)
            self.ui.result_box.setEnabled(True)
            self.ui.mel_2.setEnabled(True)
            self.ui.bk_2.setEnabled(True)
            self.ui.nv_2.setEnabled(True)
            self.ui.df_2.setEnabled(True)
            self.ui.bcc_2.setEnabled(True)        
            #self.ui.vasc_2.setEnabled(True)             
            #self.ui.ak_2.setEnabled(True)        
            #self.ui.scc_2.setEnabled(True)             
            ##self.ui.result_lable.setEnabled(True)        
            self.ui.time1.setEnabled(True) 
            self.ui.time2.setEnabled(True) 
              
# =============================================================================
#             end2222222222222222222
# =============================================================================
if __name__=="__main__":    
    app = QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())