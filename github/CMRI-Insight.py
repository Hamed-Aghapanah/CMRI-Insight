"""   Created on Sun Apr  7 12 : 04 : 55 2024

@author        :    Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation   :   Isfahan University of Medical Sciences

"""


import datetime
from time import gmtime, strftime
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
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog
from PyQt5.uic import loadUi  # Import loadUi to load UI from .ui file
from PyQt5 import QtCore
import pydicom
import nibabel as nib

# =============================================================================
# warrning off
# ==============lo===============================================================
import warnings
import sys
# import shutup; shutup.please()
if not sys.warnoptions : 
    warnings.simplefilter("ignore")
def fxn() : warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings() : 
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings("ignore")
warnings.warn('my warning')
with warnings.catch_warnings() : 
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
def fxn() : 
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings() : 
    warnings.simplefilter("ignore")
    fxn()
# =============================================================================
#  init
# =============================================================================
SALSA=False
localizer=False


SALSA=True
localizer=True

# =============================================================================
# function
# =============================================================================
import pydicom

def dicom_to_list(dicom_file_path) : 
    dicom_data = pydicom.dcmread(dicom_file_path)
    tag_value_list = []
    for elem in dicom_data.iterall() : 
        tag_keyword = elem.name
        tag_value = str(elem.value)
        tag_value_list.append((tag_keyword, tag_value))
    return tag_value_list

from keras.models import load_model

def load_cardsegnet_model():
    try:
        import os
        current_directory = os.getcwd()
        print('CardSegNet is loading ...')
        weights_directory = os.path.join(current_directory, 'weights')  # Path to the weights directory
        w_cardseg= weights_directory+ '\CardSegNet.hdf5'
        if os.path.exists(w_cardseg):
            model = load_model(w_cardseg)
            print("CardSegNet model loaded successfully.")
            return model
    except Exception as e:
        print(f"1 Error loading CardSegNet model: {e}")
        return None

def load_mecardnet_model():
    try:
        import os
        current_directory = os.getcwd()
        print('mecardnet is loading ...')
        weights_directory = os.path.join(current_directory, 'weights')  # Path to the weights directory
        w_cardseg= weights_directory+ '\MECardNet.hdf5'
        if os.path.exists(w_cardseg):
            model = load_model(w_cardseg)
            print("mecardnet model loaded successfully.")
            return model
    except Exception as e:
        print(f"2 Error loading mecardnet model: {e}")
        return None

from PyQt5.QtCore import Qt
class MyForm(QMainWindow) : 
    
    def on_cardsegnet_triggered(self):
        model = load_cardsegnet_model()
        if model:
            # self.perform_segmentation(model)
            print('CardSegNet is loaded')
            
    def on_mecardnet_triggered(self):
        model = load_mecardnet_model()
        if model:
            # self.perform_segmentation(model)
            print('mecardnet is loaded')        
            
    def __init__(self) : 
        super().__init__()
         
        self.activateWindow()
        self.raise_()
        # self.setWindowFlag(Qt.FramelessWindowHint)
        self.setGeometry(50, 50, 600, 400)
        
        self.initUI()
        # self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.SA_NO =0
        self.LA_NO =0
        self.all_SA_slice=[]
        # self.setGeometry(960, 490, 471, 441)
        self.orientations = []
        
        self.actionread_Path.triggered.connect(self.get_path_clicked)
        
        
        # xxxx
        self.Preview.clicked.connect(self.Preview_function)
        # self.Preview.triggered.connect(self.Preview_function)
         
        
        self.actionNew_Project.triggered.connect(self.get_path_clicked)
        
        
        self.progressBar.setValue(0)
        self.progressBar_2.setValue(0)
        
        self.progressBar.setStyleSheet("QProgressBar {"
                                "border :  2px solid red;"
                                "border-radius :  5px;"
                                "background-color :  black;"
                                "color :  yellow;"  # Set text color to yellow
                                "text-align :  center;"  # Center-align text
                                "}"
                                "QProgressBar :  : chunk {"
                                "background-color :  blue;"
                                "}")

        T = ['Protocol','dim2','thick2','pix2','space3', 'Modality2',    'info_image','ID2','Age2','SEX2', 'NAME2' , 'frame_no' , 'slice_no', 'SALA','checkBox','checkBox_2','private_creator','Institution_Name','seriesdescription','file_name']
       
        for i in T : 
            element = getattr(self, i)
            element.setStyleSheet("background-color :  rgb(160, 160, 160); font :  75 12pt 'Times New Roman'; color :  rgb(0, 0, 0);")
        \
        self.progressBar_2.setStyleSheet("QProgressBar {"
                    "border :  2px solid red;"
                    "border-radius :  5px;"
                    "background-color :  black;"
                    "color :  yellow;"  # Set text color to yellow
                    "text-align :  center;"  # Center-align text
                    "}"
                    "QProgressBar :  : chunk {"
                    "background-color :  blue;"
                    "}")
        self.current_frame=0
        self.current_slice=0
        self.all_frames=0
        self.all_slices=0
        self.checkBox.setEnabled(True)
        self.checkBox1=False
        self.checkBox_2.setEnabled(True)
        self.checkBox2=False
        
        self.file_name.setText('File name :  ')
        self.orientations=[]
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        # from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
        # self.canvas = FigureCanvas(self.fig5)
        # layout.addWidget(self.canvas)
        # self.plot_surfaces()
        
        
        
        print('initialized variaables ')
        
    def initUI(self) : 
        
        
        # Load UI from the generated file
        loadUi('UI.ui', self) 
        
        
        self.menuAutomatic.triggered.connect(self.on_cardsegnet_triggered)
        self.menuAutomatic.triggered.connect(self.on_mecardnet_triggered)

        
                
                
                
        self.setWindowTitle("  CMRI Insight :  A GUI based Open-Source Segmentation and Motion Tracking Application  ver 2.1.4")
        # Connect get_path_button to get_path_clicked method
        self.get_path_button.setEnabled(True)
        
        self.get_path_button.clicked.connect(self.get_path_clicked)
        
        self.NF.clicked.connect(lambda :  self.SF_clicked(0, 1))
        self.PF.clicked.connect(lambda :  self.SF_clicked(0, -1))
        self.NS.clicked.connect(lambda :  self.SF_clicked(1, 0))
        self.PS.clicked.connect(lambda :  self.SF_clicked(-1, 0))
        
        
        self.creat_contour.clicked.connect(self.creat_contour_function)
        self.load_contour.clicked.connect(self.load_contour_function)
        self.Edit_contour.clicked.connect(self.Edit_contour_function)
        
 # =============================================================================
 # showing  localizer
 # =============================================================================
    # def plot_surfaces(self) : 
    #     # ax = self.fig5.add_subplot(111, projection='3d')
    #     x = np.linspace(-5, 5, 100)
    #     y = np.linspace(-5, 5, 100)
    #     x, y = np.meshgrid(x, y)

    #     for i, orientation in enumerate(self.orientations) : 
    #         z = orientation[0] * x + orientation[1] * y + orientation[2]
    #         if i ==  self.current_slice : 
    #             ax.plot_surface(x, y, z, cmap='viridis', alpha=0.5)  # Highlight the last plane
    #         else : 
    #             ax.plot_surface(x, y, z, color='gray')

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     self.fig5.draw()
    def localizer_draw(self, slice1) : 
        
        s=1
# =============================================================================
#        showing function 
# =============================================================================
    def dis1(self, image0) : 
        self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        normalized_img = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the normalized image to uint8
        uint8_resized = normalized_img.astype(np.uint8)
        # Convert to RGB
        ResultImage = cv2.cvtColor(uint8_resized, cv2.COLOR_BGR2RGB)
        # Create QImage
        height, width, channel = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # Convert QImage to QPixmap
        pixmap = QtGui.QPixmap.fromImage(QImg)
        # Create QGraphicsScene and add the pixmap
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.fig1.setScene(scene)

    def dis2(self, image0) : 
        self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        normalized_img = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the normalized image to uint8
        uint8_resized = normalized_img.astype(np.uint8)
        # Convert to RGB
        ResultImage = cv2.cvtColor(uint8_resized, cv2.COLOR_BGR2RGB)
        # Create QImage
        height, width, channel = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # Convert QImage to QPixmap
        pixmap = QtGui.QPixmap.fromImage(QImg)
        # Create QGraphicsScene and add the pixmap
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.fig2.setScene(scene)
        
    def dis3(self, image0) : 
        self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        normalized_img = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the normalized image to uint8
        uint8_resized = normalized_img.astype(np.uint8)
        # Convert to RGB
        ResultImage = cv2.cvtColor(uint8_resized, cv2.COLOR_BGR2RGB)
        # Create QImage
        height, width, channel = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # Convert QImage to QPixmap
        pixmap = QtGui.QPixmap.fromImage(QImg)
        # Create QGraphicsScene and add the pixmap
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.fig3.setScene(scene)
        
    def dis4(self, image0) : 
        self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        normalized_img = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the normalized image to uint8
        uint8_resized = normalized_img.astype(np.uint8)
        # Convert to RGB
        ResultImage = cv2.cvtColor(uint8_resized, cv2.COLOR_BGR2RGB)
        # Create QImage
        height, width, channel = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # Convert QImage to QPixmap
        pixmap = QtGui.QPixmap.fromImage(QImg)
        # Create QGraphicsScene and add the pixmap
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.fig4.setScene(scene)
 
    def dis5(self, image0) : 
        self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 320
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        normalized_img = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the normalized image to uint8
        uint8_resized = normalized_img.astype(np.uint8)
        # Convert to RGB
        ResultImage = cv2.cvtColor(uint8_resized, cv2.COLOR_BGR2RGB)
        # Create QImage
        height, width, channel = ResultImage.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(ResultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # Convert QImage to QPixmap
        pixmap = QtGui.QPixmap.fromImage(QImg)
        # Create QGraphicsScene and add the pixmap
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.fig5.setScene(scene)
 # =============================================================================
 # function moving around heart 
 # =============================================================================
    def SF_clicked(self,S,F ) : 
        print ('self.loading =',self.loading)
        self.load_contour_function()
                
        if not self.loading  :  
            image0=np.zeros([100,100])
            self.dis2( image0);self.dis3( image0);self.dis4( image0)
            
        
        import cv2
        if F==1 : 
            self.NF.setStyleSheet("background-color :  red;")
            if self.current_frame <self.all_frames-1 : 
                self.current_frame=self.current_frame +1
            import cv2    
            cv2.waitKey(100)
            self.NF.setStyleSheet("background-color :  rgb(200,200, 150);")
        if F==-1 : 
            self.PF.setStyleSheet("background-color :  red;")
            if self.current_frame >0 : 
                self.current_frame=self.current_frame -1
                
            cv2.waitKey(100)
            self.PF.setStyleSheet("background-color :  rgb(150,200, 255);")
        if S==1 : 
            self.NS.setStyleSheet("background-color :  red;")
            if self.current_slice <self.all_slices-1 : 
                self.current_slice=self.current_slice +1
            cv2.waitKey(100)
            self.NS.setStyleSheet("background-color :  rgb(200,200, 150);")
        if S==-1 : 
            self.PS.setStyleSheet("background-color :  red;")
            if self.current_slice >0 : 
                self.current_slice=self.current_slice -1
            cv2.waitKey(100)
            self.PS.setStyleSheet("background-color :  rgb(150,200, 255);") 
        
        self.NS_2.setText( str(self.current_slice+1) )
        self.NF_2.setText( str(self.current_frame+1) )
        
        print('current_slice =',self.current_slice, ' , current_frame =',self.current_frame)
        self.load_contour_function()
        dicom_data1 = self.dicom_files [self.current_slice]
        import pydicom
        dicom_data = pydicom.dcmread(dicom_data1)
        image = dicom_data.pixel_array
        self.image = np.uint8(image[self.current_frame , : , : ])
        image0=image
        try : 
            print('np.shape(image) =',np.shape(image))
            if np.size(np.shape(image))>2 : 
                image0=image[self.current_frame ,  : , : ]
                currnet_image=image0
            self.currnet_image= currnet_image
            print('np.shape(image) =',np.shape(image))
        except : 1
        self.dis1(image0) 
        
        import pydicom
        dicom_data = pydicom.dcmread(self.dicom_files [self.current_slice])
        patient_id = dicom_data.get("PatientID", "N/A")
        
        # dicom_data = pydicom.dcmread(file_path)
        patient_id = dicom_data.PatientID
        patient_name = dicom_data.PatientName 
        SA_NO=0
        LA_NO=0
        
        dicom_file_path=self.dicom_files [self.current_slice]
 
        tags_and_values = dicom_to_list(dicom_file_path)
        private_creator = None;orientation = None; inst11 = None; seriesdescription=None
        sex=None;         modality='MRI'
        manufacturer=None;    pat_name=None;         pat_ID=None;         code_meaning=None
        pat_size=None;  pat_weight=None;     pat_b_d=None;address=None; 
        date_pat=None;date_study=None;Protocol=None;tthick2=None;ppix2=None  ;sspace3=None 
        
        for tag, value in tags_and_values : 
            keyword=tag
            
            # saedsaed2
            
            if 'Spacing'.lower() in tag.lower()  and 'Slices'.lower() in tag.lower()  :  
                sspace3=value
            
            if 'date'.lower() in tag.lower()  and 'pat'.lower() in tag.lower()  :  
                date_pat=value
            if 'date'.lower() in tag.lower()  and 'Study'.lower() in tag.lower()  :  
                date_study=value
                
            if 'Protocol'.lower() in tag.lower()  and 'Name'.lower() in tag.lower()  : 
                Protocol = value
                
            if 'Slice'.lower() in keyword.lower()  and 'Thickness'.lower() in keyword.lower()  : 
                tthick2=value
            if 'Pixel'.lower() in keyword.lower()  and 'Spacing'.lower() in keyword.lower()  : 
                ppix2=value 
            
            if 'manu'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                private_creator=value
            if 'pat'.lower() in keyword.lower()  and 'Orient'.lower() in keyword.lower()  : 
                orientation = value
                
            if 'series'.lower() in keyword.lower()  and 'description'.lower() in keyword.lower()  : 
                seriesdescription = value
                
            if 'sex'.lower() in keyword.lower()  and 'pat'.lower() in keyword.lower()  : 
                sex = value
                if 'M' in value: sex='Male'
                if 'F' in value: sex='Female'
            if 'inst'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                inst = value
            if 'add'.lower() in keyword.lower()  and 'Institution'.lower() in keyword.lower()  : 
                address = value    
            if 'manufacturer'.lower() in keyword.lower()  and 'model'.lower() in keyword.lower()  : 
                manufacturer = value
            if 'pat'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                pat_name = value
            if  'patient id'.lower() == keyword.lower()    : 
                pat_ID = value
            if  'Modality'.lower() == keyword.lower()    : 
                modality = value
            if 'pat'.lower() in keyword.lower()  and 'birt'.lower() in keyword.lower() and 'date'.lower() in keyword.lower()  : 
                pat_b_d = value
                from datetime import datetime
                date_obj = datetime.strptime(pat_b_d, "%Y%m%d")
                year = str(date_obj.year)
                month = str(date_obj.month)
                day = str(date_obj.day)
                pat_b_d=year+' : '+month+' : '+day
        
            if 'pat'.lower() in keyword.lower()  and 'size'.lower() in keyword.lower()  : 
                pat_size = value    
            if 'pat'.lower() in keyword.lower()  and 'weight'.lower() in keyword.lower()  : 
                pat_weight = value
            if 'code'.lower() in keyword.lower()  and 'mean'.lower() in keyword.lower()  : 
                code_meaning = value  
            if 'Institution'.lower() in keyword.lower()   and 'name'.lower() in keyword.lower() and not 'dep'.lower() in keyword.lower()   : 
                inst11 = value
            
            # if 'Protocol'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
            #     Protocol = value
            #     print(1, '  ',Protocol ,keyword.lower())
                
        patient_name=pat_name
        try : 
            parts = patient_name.split('^'); patient_name = parts[1] + ' ' + parts[0]
        except : 1
        
        # Protocol = dicom_file.get('SeriesDescription', 'No Protocol Name Available')
        # print(10*'W')
        
        try :         age=str (round (  int(date_study)/10000 - int(date_pat)/10000)  )
        except : age=' 53'

        # dimm=str(self.image_pixel )
        dimm=str(np.shape(self.image_pixel))
        # dimm=' 256*256'
        # tthick2='6 mm'
        # ppix2=' 6 mm'
        # sspace3=' 5 mm'
        Protocol ='T1WI'
        print('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))
        self.ID2.setText('ID :  '+ str(patient_id))
        self.NAME2.setText('NAME :  '+ str(patient_name))
        self.SEX2.setText('SEX : '+sex)
        self.Age2.setText('Age : '+age)
        
        self.slice_no.setText('Slice No :  '+ str(self.all_slices))
        self.frame_no.setText('Frame No :  '+ str(self.all_frames))
        self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))
        self.data.setText('Orientation :  '+str(orientation))
        self.private_creator.setText('Private Creator : '+str(private_creator))
        self.Modality2.setText(' Modality : '+str(modality))
        address2=''
        for i in address:
            if i.isalpha():address2=address2 + i
            else:address2=address2 +  ' '
        self.Institution_Name.setText('Institution Name :  '+str(inst11)[:-2]+' , '+ str (address2))
        self.seriesdescription.setText('Series Description :  '+str(seriesdescription))
        
        self.dim2.setText('Image Dimensions : '+str(dimm))
        self.thick2.setText('Slice Thickness : '+str(tthick2))
        self.pix2.setText('Pixel Spacing : '+str(ppix2))
        self.space3.setText('Spacing Between Slices : '+str(sspace3))
        self.Protocol.setText('Protocol  : '+str(Protocol))
        
        
        
        self.seriesdescription1=str(seriesdescription)
        import os
        filename = os.path.basename(self.dicom_files [self.current_slice])
        
        self.file_name.setText('File name :  '+filename)
        
        # self.current_frame =0
        # self.current_slice =0
        image0=image
        if np.size(np.shape(image))>2 : 
            image0=image[self.current_frame ,  : , : ]
        self.dis1(image0)
        self.load_contour_function()
        # print(10*'s')
        print (self.all_SA_slice)
        print (self.current_slice)
        if 'sa' in self.seriesdescription1.lower()  and   np.abs (S)>0.5 and self.checkBox and self.checkBox2  : 
            from localizer_draw import localizer_draw
            current_slice=self.current_slice
            for i in range (len (self.all_SA_slice)) : 
                if current_slice ==self.all_SA_slice [i] : 
                    current_slice_sa = i
                    
            all_SA_slices = len (self.all_SA_slice)
            image_localizer = localizer_draw (current_slice_sa ,  all_SA_slices)
            self.dis5( image_localizer);
        
        import cv2
        cv2.waitKey(10)
        cnt=0
# =============================================================================
# get path function
# =============================================================================
    def checkbox_state_changed(self, state) : 
        if state == 2 :   # Qt.Checked
            print("Checkbox is checked")
            self.checkBox1=True
        else : 
            self.checkBox1=False
            print("Checkbox is unchecked")
    def checkbox_state_changed2(self, state) : 
        if state == 2 :   # Qt.Checked
            print("Checkbox2 is checked")
            self.checkBox2=True
        else : 
            self.checkBox2=False
            print("Checkbox2 is unchecked")
# =============================================================================
# creat_contour   clicked
# =============================================================================     
    def creat_contour_function(self) : 
        from manual_segmentation_function import segmentator
        # currnet_image=self.currnet_image 
        # print ( 'np.shape (self.image_pixel)  :  ' ,np.shape (self.image_pixel))
        # currnet_image= self.image_pixel [self.current_frame, : , :  ]
        
        
        dicom_data1 = self.dicom_files [self.current_slice]
        import pydicom
        dicom_data = pydicom.dcmread(dicom_data1)
        image = dicom_data.pixel_array
        self.image_pixel = image
        print ( 'np.shape (self.image_pixel)  :  ' ,np.shape (self.image_pixel))
        currnet_image= self.image_pixel [self.current_frame, : , :  ]
        
        
        # currnet_image=image[self.current_frame ,  : , : ]
        
        try : 
            print('np.shape(currnet_image)1 =',np.shape(currnet_image))
            if np.size(np.shape(currnet_image))>2 :  currnet_image=currnet_image[self.current_frame ,  : , : ]
            self.currnet_image= currnet_image 
            print('np.shape(currnet_image)2 =',np.shape(currnet_image))
        except : 1
        
        
        print('np.shape(currnet_image)3=',np.shape(currnet_image))
        # self.current_slic
        i=self.current_frame
        try : 
            saved_image, mask_r  ,mask_l , mask_m ,point_r, point_l,  point_m = segmentator (currnet_image,i)
            print('np.shape(saved_image)',np.shape(saved_image))
            try : 
                rgb_image = cv2.cvtColor(saved_image, cv2.COLOR_GRAY2RGB)
            except : 
                rgb_image = saved_image
            print('s==0')
            mask_l3=0*rgb_image;mask_l3[ : , : ,0]=mask_l
            print('s==1')
            mask_r3=0*rgb_image;mask_r3[ : , : ,2]=mask_r
            mask_m3=0*rgb_image;mask_m3[ : , : ,1]=mask_m
            print('s==2')
            result_r=cv2.addWeighted(rgb_image, 1, mask_r3, 0.3, 0)
            result_l=cv2.addWeighted(rgb_image, 1, mask_l3, 0.3, 0)
            result_m=cv2.addWeighted(rgb_image, 1, mask_m3, 0.3, 0)
            color1=(0, 0, 255)
            color3=(255, 0, 0)
            color2 =(0, 255, 255)
            
            for x,y in point_r  : 
                result_r = cv2.circle(result_r, (x, y), 1, color1, 1)
            for x,y in point_l  : 
                result_l= cv2.circle(result_l, (x, y), 1, color2, 1)
            for x,y in point_m  : 
                result_m = cv2.circle(result_m, (x, y), 1, color3, 1)
                
                
            print('s=3')
            self.dis3(result_r)
            self.dis2(result_l)
            self.dis4(result_m)
            print('s=4')
            
            
            
            id_patient = self.patient_id
            import os 
            
            # save 4d images
            self.path_temp = self.path+'//temp'
            try : os.mkdir(self.path_temp)
            except : pass
            
            p1=self.path_temp+'// patient' +str(id_patient)+'frame'+str(self.current_frame)+'slice'+str(self.current_slice)+'.npz'
            # saved_image, mask_r  ,mask_l , mask_m ,point_r, point_l,  point_m
            
            mask_r = np.array(mask_r)                                                                                 
            mask_l = np.array(mask_l)                                                                                 
            mask_m = np.array(mask_m)  
                                                                                   
            point_r = np.array(point_r)                                                                                 
            point_l = np.array(point_l)                                                                                 
            point_m = np.array(point_m)   
            saved_image = np.array(saved_image)                                                                               
    
            np.savez(p1, saved_image=saved_image, mask_r=mask_r,mask_m=mask_m,mask_l=mask_l,point_r=point_r,point_l=point_l,point_m=point_m,)
    # =============================================================================
    #         saed
    # =============================================================================
            # nifti_data = np.stack([saved_image, mask_r  ,mask_l , mask_m ], axis=-1)
            # nifti_image = nib.Nifti1Image(nifti_data, affine=np.eye(4))
            # p1=self.path_temp+'// patient' +str(id_patient)+'frame'+str(self.current_frame)+'slice'+str(self.current_slice+'.nii.gz'
            # nib.save(nifti_image, p1)
    
            # fig4 Myo
        except : pass
 # =============================================================================
 # load_contour   clicked
 # =============================================================================   


    def Edit_contour_function   (self) : 
        print('Edit_contour_function is not completed')
        
        
    def load_contour_function(self) : 
        self.loading=False
        try : 
            print ('load_contour_function')
            # print('loading')
            import os
            self.path_temp = self.path+'//temp'
            
            
            # try : os.mkdir(self.path_temp)
            # except : pass
            
            import os

            if not os.path.exists(self.path_temp) : 
                os.mkdir(self.path_temp)
            
            id_patient = self.patient_id
            p1=self.path_temp+'// patient' +str(id_patient)+'frame'+str(self.current_frame)+'slice'+str(self.current_slice)+'.npz'
            # print('loading1')
            print ('p1 =' ,p1)
            if  not os.path.exists(p1) : 
                self.loading=False
                self.dis3(np.zeros ([100,100]))
                self.dis2(np.zeros ([100,100]))
                self.dis4(np.zeros ([100,100]))
                 
                
                
            if  os.path.exists(p1) : 
                loaded_arrays = np.load(p1)
                mask_r = loaded_arrays['mask_r']
                mask_l = loaded_arrays['mask_l']
                mask_m = loaded_arrays['mask_m']
                # print('loading2')
                point_r = loaded_arrays['point_r']
                point_l = loaded_arrays['point_l']
                point_m = loaded_arrays['point_m']
                
                saved_image= loaded_arrays['saved_image']
                
                print ('np.shape (saved_image)',np.shape (saved_image) )
                rgb_image = saved_image
                print('loading3')
                
                mask_l3=0*rgb_image;mask_l3[ : , : ,0]=mask_l
                mask_r3=0*rgb_image;mask_r3[ : , : ,2]=mask_r
                mask_m3=0*rgb_image;mask_m3[ : , : ,1]=mask_m
                result_r=cv2.addWeighted(rgb_image, 1, mask_r3, 0.3, 0)
                result_l=cv2.addWeighted(rgb_image, 1, mask_l3, 0.3, 0)
                result_m=cv2.addWeighted(rgb_image, 1, mask_m3, 0.3, 0)
                color1=(0, 0, 255)
                color3=(255, 0, 0)
                color2 =(0, 255, 255)
                for x,y in point_r  : 
                    result_r = cv2.circle(result_r, (x, y), 1, color1, 1)
                for x,y in point_l  : 
                    result_l= cv2.circle(result_l, (x, y), 1, color2, 1)
                for x,y in point_m  : 
                    result_m = cv2.circle(result_m, (x, y), 1, color3, 1)
                    
               
                self.dis3(result_r)
                self.dis2(result_l)
                self.dis4(result_m)
                
                self.loading=True
                print ('self.loading =' ,self.loading)
        except : 
            self.loading=False
            
            print('can not load load_contour_function') 
                 
    def Preview_function(self) : 
        
        all_SA_slice=[]
        print('if self.checkBox1 and SALSA')
        print( self.checkBox1 , SALSA)
        cnt=0;LA_NO = 0;SA_NO = 0
        if self.checkBox1 and SALSA : 
            # sss
            dicom_files=self.dicom_files
            for dicom_data1 in dicom_files : 
                cnt=cnt+1
                a=int (np. round (100*cnt/len(dicom_files)))
                self.progressBar.setValue(a)
                
                self.progressBar.setStyleSheet("QProgressBar {"
                            "border :  2px solid red;"
                            "border-radius :  5px;"
                            "background-color :  black;"
                            "color :  yellow;"  # Set text color to yellow
                            "text-align :  center;"  # Center-align text
                            "}"
                            "QProgressBar :  : chunk {"
                            "background-color :  blue;"
                            "}")
                
                dicom_file_path = dicom_data1
                tags_and_values = dicom_to_list(dicom_file_path)
                for tag, value in tags_and_values : 
                    if 'series'.lower() in tag.lower()  and 'description'.lower() in tag.lower()  : 
                        seriesdescription = value
                        t=0
                        if 'LA'.lower()in value.lower() or '2ch'in value.lower() or '3ch'.lower()in value.lower() or '4ch'in value.lower() : 
                            LA_NO=LA_NO+1;t=1
                            self.LA_NO =LA_NO
                        if t==0 and 'Short'.lower()in value.lower() or 'SA'.lower()in value.lower() : 
                            SA_NO=SA_NO+1;t=1
                            all_SA_slice.append (cnt -1)
                            self.SA_NO =SA_NO
        self.all_SA_slice=all_SA_slice
        self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))                 
        self.NS_2.setText( str(self.current_slice+1) )
        self.NF_2.setText( str(self.current_frame+1) )
        
        # print ('all_SA_slice' ,all_SA_slice)
         
        self.progressBar.setStyleSheet("QProgressBar {"
                            "border :  2px solid red;"
                            "border-radius :  5px;"
                            "background-color :  black;"
                            "color :  yellow;"  # Set text color to yellow
                            "text-align :  center;"  # Center-align text
                            "}"
                            "QProgressBar :  : chunk {"
                            "background-color :  green;"
                            "}")

        # self.get_path_button.clicked.connect(self.get_path_clicked)
# =============================================================================
#             localizer
# =============================================================================
        # self.orientations=[]
        import cv2    
        cv2.waitKey(100)
        orientations=[]
        pixel_spacing=[]
        cnt=0
        print( 'self.checkBox2 , Localizer')
        print( self.checkBox2 , localizer)
        orientations_slice=[]
        try : 
            dicom_files=self.dicom_files
            if self.checkBox2 and localizer : 
                for dicom_data1 in dicom_files : 
                    print('cnt',cnt)
                    cnt=cnt+1
                    a=int (np. round (100*cnt/len(dicom_files)))
                    self.progressBar_2.setValue(a)
                    
                    self.progressBar_2.setStyleSheet("QProgressBar {"
                                "border :  2px solid red;"
                                "border-radius :  5px;"
                                "background-color :  black;"
                                "color :  yellow;"  # Set text color to yellow
                                "text-align :  center;"  # Center-align text
                                "}"
                                "QProgressBar :  : chunk {"
                                "background-color :  blue;"
                                "}")
                    dicom_file_path = dicom_data1
                    tags_and_values = dicom_to_list(dicom_file_path)
                    # print(dicom_data1,'np.shape(dicom_file_path) =', np.shape(dicom_file_path))
                    
                    cc=0
                    for tag, value in tags_and_values : 
                        keyword=tag
                        cc=cc+1
                        # if cc<80 : 
                        #     print(40*'+');print(tag,value)
                        if 'pixel'.lower() in keyword.lower()  and 'space'.lower() in keyword.lower()  : 
                             orientation = value
                             import ast
                             number_vector = ast.literal_eval(orientation)
                             pixel_spacing.append(number_vector)
                             print(cnt, 'pixel_spacing =',number_vector)
                             
                        if 'pat'.lower() in keyword.lower()  and 'Orient'.lower() in keyword.lower()  : 
                             orientation = value
                             import ast
                             number_vector = ast.literal_eval(orientation)
                             orientations.append(number_vector)
                             # print(cnt, 'orientation =',number_vector)
                             orientations_slice.append(cnt-1)
                             # break
                
                # np.save(directory_path +'/orientations.npy', orientations) 
                # np.save(directory_path +'/pixel_spacing.npy', pixel_spacing) 
                self.orientations=orientations
                self.orientations_slice=orientations_slice
                
                self.progressBar_2.setStyleSheet("QProgressBar {"
                            "border :  2px solid red;"
                            "border-radius :  5px;"
                            "background-color :  black;"
                            "color :  yellow;"  # Set text color to yellow
                            "text-align :  center;"  # Center-align text
                            "}"
                            "QProgressBar :  : chunk {"
                            "background-color :  green;"
                            "}")
                
                
                # if 'sa' in self.filename : 
                if 'sa' in self.seriesdescription1.lower()     and self.checkBox and self.checkBox2  : 
                    from localizer_draw import localizer_draw
                    current_slice=self.current_slice
                    for i in range (len (self.all_SA_slice)) : 
                        if current_slice ==self.all_SA_slice [i] : 
                            current_slice_sa = i
                            
                    all_SA_slices = len (self.all_SA_slice)
                    image_localizer = localizer_draw (current_slice_sa ,  all_SA_slices)
                    self.dis5( image_localizer);
                    
                print('localizer is finish' )
        except : pass
      # import numpy as np       
# =============================================================================
# get path clicked
# =============================================================================     
    def get_path_clicked(self) : 
        import numpy as np
        self.NF.setEnabled(True)
        self.PF.setEnabled(True)
        self.NS.setEnabled(True)
        self.PS.setEnabled(True)
        
        self.creat_contour.setEnabled(True)
        self.load_contour.setEnabled(True)
        
        self.creat_contour.setStyleSheet("background-color :  rgb(150,250, 150);")
        self.load_contour.setStyleSheet("background-color :  rgb(150,250, 150);")
        self.Edit_contour.setStyleSheet("background-color :  rgb(150,250, 150);")
        
        
        
        directory_path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        
        import os

        # Function to list DICOM files
        def list_dicom_files(directory_path):
            dicom_files = []
            for dirpath, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    if filename.endswith('.dcm'):
                        dicom_files.append(os.path.join(dirpath, filename))
            return dicom_files
        
        # Main logic for directory selection and processing
        # directory_path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        self.path = str(directory_path)
        
        try:
            split_parts = self.path.split('viewer/')
            print('self.path = ', self.path)
        
            after_viewer = split_parts[1]
        except:
            after_viewer = self.path
        
        # Define the condition for A
        dicom_files = list_dicom_files(directory_path)
        A = len(dicom_files) > 0  


        self.path= str(directory_path)
        try:
            split_parts = self.path.split('viewer/')
            print('self.path = ', self.path)
    
            after_viewer = split_parts[1]
        except:
            after_viewer = self.path
        
        
        
        
        
        if A:
            if  not 'DICOM' in after_viewer or not 'dicom' in after_viewer : 
                self.path=self.path+'//DICOM'
                # print('saed2')
            print(self.path)
            print('saed3')
            print ('directory_path ',self.path)
            self.directory_path=directory_path
            if directory_path : 
                # print("Selected directory : ", directory_path)
                import os 
                def list_dicom_files(directory_path) : 
                    dicom_files = []
                    for dirpath, _, filenames in os.walk(directory_path) : 
                        for filename in filenames : 
                            if filename.endswith('.dcm') : 
                                dicom_files.append(os.path.join(dirpath, filename))
                    return dicom_files
    
                # current_directory = os.getcwd()
                dicom_files = list_dicom_files(directory_path)
                self.dicom_files=dicom_files
                # print(dicom_files)
                import pydicom
                dicom_data = pydicom.dcmread(dicom_files[0])
                patient_id = dicom_data.get("PatientID", "N/A")
                self.patient_id=patient_id
                # dicom_data = pydicom.dcmread(file_path)
                patient_id = dicom_data.PatientID
                patient_name = dicom_data.PatientName 
                SA_NO=0
                LA_NO=0                  
    
                for dicom_data1 in dicom_files : 
                    dicom_data = pydicom.dcmread(dicom_data1)
                    image = dicom_data.pixel_array
                    
                    if np.size(np.shape(image))>2 : 
                        frame_no001=np.shape(image)[0]
                        dicom_dict = []
                        # cnt = 0
                        for data_element in dicom_data : 
                            # cnt=cnt+1
                            tag = data_element.tag
                            value = data_element.value
                            dicom_dict.append({'tag' :  tag, 'value' :  value})
                        # dicom_file_path = r"F : \0001phd\00_thesis\0_mfiles\1_local_dataset\01_dicom viewer\01_DICOM_viewer\1705364\DICOM\series0007-Body\img0001MultiFrame25-unknown.dcm"
                        dicom_file_path = dicom_data1
                        tags_and_values = dicom_to_list(dicom_file_path)
                        private_creator = None;orientation = None; inst11 = None; seriesdescription=None
                        sex=None;         modality=' MR'
                        manufacturer=None;    pat_name=None;         pat_ID=None;         code_meaning=None
                        pat_size=None;  pat_weight=None;     pat_b_d=None;address=None; 
                        date_pat=None;date_study=None;Protocol=None;tthick2=None;ppix2=None;sspace3=None 
                        
                        # CNT=0
                        for tag, value in tags_and_values : 
                            # CNT=CNT+1
                            # print(CNT,' ', tag)
                            keyword=tag
                            # saedsaed
                            if 'Spacing'.lower() in tag.lower()  and 'Slices'.lower() in tag.lower()  :  
                                sspace3=value
                            if 'date'.lower() in tag.lower()  and 'pat'.lower() in tag.lower()  :  
                                date_pat=value
                            if 'date'.lower() in tag.lower()  and 'Study'.lower() in tag.lower()  :  
                                date_study=value
                            # if 'Description'.lower() in tag.lower() or   'Description'.lower() in value.lower()  : 
                            #     print(10*'\n')
                            #     print(tag)
                            #     print(value)
                                 
                                
                            if 'Protocol'.lower() in tag.lower()  and 'Name'.lower() in tag.lower()  : 
                                Protocol = value
                                
                            if 'Slice'.lower() in keyword.lower()  and 'Thickness'.lower() in keyword.lower()  : 
                                tthick2=value
                            if 'Pixel'.lower() in keyword.lower()  and 'Spacing'.lower() in keyword.lower()  : 
                                ppix2=value    
                            
                                
                            
                            
                            
                            if 'manu'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                                private_creator=value
                                
                            if 'pat'.lower() in keyword.lower()  and 'Orient'.lower() in keyword.lower()  : 
                                orientation = value
                                # print(orientation)
                                
                            if 'series'.lower() in keyword.lower()  and 'description'.lower() in keyword.lower()  : 
                                seriesdescription = value
                                
                            if 'sex'.lower() in keyword.lower()  and 'pat'.lower() in keyword.lower()  : 
                                sex = value
                                if 'M' in value: sex='Male'
                                if 'F' in value: sex='Female'
                                
                                
                            if 'inst'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                                inst = value
                            if 'add'.lower() in keyword.lower()  and 'Institution'.lower() in keyword.lower()  : 
                                address = value    
                            if 'manufacturer'.lower() in keyword.lower()  and 'model'.lower() in keyword.lower()  : 
                                manufacturer = value
                            if 'pat'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                                pat_name = value
                            if  'patient id'.lower() == keyword.lower()    : 
                                pat_ID = value
                            if  'modal'.lower() == keyword.lower()    : 
                                modality = value
                            if 'pat'.lower() in keyword.lower()  and 'birt'.lower() in keyword.lower() and 'date'.lower() in keyword.lower()  : 
                                pat_b_d = value
                                from datetime import datetime
                                date_obj = datetime.strptime(pat_b_d, "%Y%m%d")
                                year = str(date_obj.year)
                                month = str(date_obj.month)
                                day = str(date_obj.day)
                                pat_b_d=year+' : '+month+' : '+day
                        
                            if 'pat'.lower() in keyword.lower()  and 'size'.lower() in keyword.lower()  : 
                                pat_size = value    
                            if 'pat'.lower() in keyword.lower()  and 'weight'.lower() in keyword.lower()  : 
                                pat_weight = value
                            if 'code'.lower() in keyword.lower()  and 'mean'.lower() in keyword.lower()  : 
                                code_meaning = value  
                            if 'Institution'.lower() in keyword.lower()   and 'name'.lower() in keyword.lower() and not 'dep'.lower() in keyword.lower()   : 
                                inst11 = value
                            
                            # if 'Protocol'.lower() in keyword.lower()  and 'name'.lower() in keyword.lower()  : 
                            
                            #     print(1, '  ',Protocol ,keyword.lower())
                                
                            # if 'age'.lower() in keyword.lower()    : 
                            #     age = value
                            #     print(1, '  ',age ,keyword.lower())
                                
                        break
                
                dicom_data1 = self.dicom_files [self.current_slice]
                import pydicom
                dicom_data = pydicom.dcmread(dicom_data1)
                image = dicom_data.pixel_array
                self.image_pixel = image
                print ('np.shape (self.image_pixel)  = ',np.shape (self.image_pixel))
                image0=image
                
                print ('RECOLOR')
                T = ['Protocol','dim2','thick2','pix2','space3', 'Modality2',    'info_image','ID2','Age2','SEX2' , 'NAME2' , 'frame_no' , 'slice_no', 'SALA','checkBox','checkBox_2','private_creator','Institution_Name','seriesdescription','file_name']
               
                for i in T : 
                    element = getattr(self, i)
                    element.setStyleSheet("background-color :  rgb(160, 160, 240); font :  75 12pt 'Times New Roman'; color :  rgb(0, 0, 0);")
                T2=['PS_3','PS_4','PS_5','PS_6',]
                for i in T2 :
                    element = getattr(self, i)
                    element.setStyleSheet("background-color :  rgb(255, 160, 160); font :  75 12pt 'Times New Roman'; color :  rgb(0, 0, 0);")
                
                    
                  
                
                patient_name=pat_name
                try : 
                    parts = patient_name.split('^'); patient_name = parts[1] + ' ' + parts[0]
                except : 1
    
                self.ID2.setText('ID :  '+ str(patient_id))
                self.NAME2.setText('NAME :  '+ str(patient_name)  )
                self.slice_no.setText('Slice No :  '+ str(len(dicom_files)))
                self.frame_no.setText('Frame No :  '+ str(frame_no001))
                self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))
                self.data.setText('Orientation :  '+str(orientation))
                self.private_creator.setText('Private Creator : '+str(private_creator))
                self.Modality2.setText(' Modality : '+str(modality))
                address2=''
                for i in address:
                    if i.isalpha():address2=address2 + i
                    else:address2=address2 +  ' '
    
                self.Institution_Name.setText('Institution Name :  '+str(inst11)+' , '+ str (address2))
                self.seriesdescription.setText('Series Description :  '+str(seriesdescription))
                
                
                # print(10*'S')
                # age=' 53'
                try :         age=str (round (  int(date_study)/10000 - int(date_pat)/10000)  )
                except : age=' 53'
                dimm=str(np.shape(self.image_pixel))
                # dimm=' 256*256'
                # tthick2='6 mm'
                # ppix2='mm'
                # sspace3=' 5 mm'
                Protocol ='T1WI'
                
                # print(ppix2)
                self.SEX2.setText('SEX : '+sex)
                self.Age2.setText('Age : '+age)
                self.dim2.setText('Image Dimensions : '+str(dimm))
                self.thick2.setText('Slice Thickness : '+str(tthick2))
                self.pix2.setText('Pixel Spacing : '+str(ppix2))
                self.space3.setText('Spacing Between Slices : '+str(sspace3))
                self.Protocol.setText('Protocol  : '+str(Protocol))
                
                
                self.all_frames=frame_no001
                self.all_slices=len(dicom_files)
                
                filename = os.path.basename(self.dicom_files [self.current_slice])
                self.seriesdescription1 =seriesdescription
                
                self.file_name.setText('File name :  '+filename)
                image0=image
                if np.size(np.shape(image))>2 : 
                    image0=image[self.current_frame ,  : , : ]
                self.dis1(image0)
                
                self.load_contour_function()
                import cv2
                cv2.waitKey(10)
                cnt=0
                
                self.checkBox.stateChanged.connect(self.checkbox_state_changed)
                self.checkBox_2.stateChanged.connect(self.checkbox_state_changed2)
                
                
                
    # =============================================================================
    #             SA  LA
    # =============================================================================
    
                print('if self.checkBox1 and SALSA')
                print( self.checkBox1 , SALSA)
                if self.checkBox1 and SALSA : 
                    self.dicom_files= dicom_files
                    for dicom_data1 in dicom_files : 
                        cnt=cnt+1
                        a=int (np. round (100*cnt/len(dicom_files)))
                        self.progressBar.setValue(a)
                        
                        self.progressBar.setStyleSheet("QProgressBar {"
                                    "border :  2px solid red;"
                                    "border-radius :  5px;"
                                    "background-color :  black;"
                                    "color :  yellow;"  # Set text color to yellow
                                    "text-align :  center;"  # Center-align text
                                    "}"
                                    "QProgressBar :  : chunk {"
                                    "background-color :  blue;"
                                    "}")
                        
                        dicom_file_path = dicom_data1
                        tags_and_values = dicom_to_list(dicom_file_path)
                        for tag, value in tags_and_values : 
                            if 'series'.lower() in tag.lower()  and 'description'.lower() in tag.lower()  : 
                                seriesdescription = value
                                t=0
                                if 'LA'.lower()in value.lower() or '2ch'in value.lower() or '3ch'.lower()in value.lower() or '4ch'in value.lower() : 
                                    LA_NO=LA_NO+1;t=1
                                    self.LA_NO =LA_NO
                                if t==0 and 'Short'.lower()in value.lower() or 'SA'.lower()in value.lower() : 
                                    SA_NO=SA_NO+1;t=1
                                    self.SA_NO =SA_NO
    
                self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))                 
                self.NS_2.setText( str(self.current_slice+1) )
                self.NF_2.setText( str(self.current_frame+1) )
                
                 
                self.progressBar.setStyleSheet("QProgressBar {"
                                    "border :  2px solid red;"
                                    "border-radius :  5px;"
                                    "background-color :  black;"
                                    "color :  yellow;"  # Set text color to yellow
                                    "text-align :  center;"  # Center-align text
                                    "}"
                                    "QProgressBar :  : chunk {"
                                    "background-color :  green;"
                                    "}")
    
                # self.get_path_button.clicked.connect(self.get_path_clicked)
    # =============================================================================
    #             localizer
    # =============================================================================
                # self.orientations=[]
                import cv2    
                cv2.waitKey(100)
                orientations=[]
                pixel_spacing=[]
                cnt=0
                print( 'self.checkBox2 , localizer')
                print( self.checkBox2 , localizer)
                orientations_slice=[]
                if self.checkBox2 and localizer : 
                    for dicom_data1 in dicom_files : 
                        print('cnt',cnt)
                        cnt=cnt+1
                        a=int (np. round (100*cnt/len(dicom_files)))
                        self.progressBar_2.setValue(a)
                        
                        self.progressBar_2.setStyleSheet("QProgressBar {"
                                    "border :  2px solid red;"
                                    "border-radius :  5px;"
                                    "background-color :  black;"
                                    "color :  yellow;"  # Set text color to yellow
                                    "text-align :  center;"  # Center-align text
                                    "}"
                                    "QProgressBar :  : chunk {"
                                    "background-color :  blue;"
                                    "}")
                        dicom_file_path = dicom_data1
                        tags_and_values = dicom_to_list(dicom_file_path)
                        # print(dicom_data1,'np.shape(dicom_file_path) =', np.shape(dicom_file_path))
                        
                        cc=0
                        for tag, value in tags_and_values : 
                            keyword=tag
                            cc=cc+1
                            # if cc<80 : 
                            #     print(40*'+');print(tag,value)
                            if 'pixel'.lower() in keyword.lower()  and 'space'.lower() in keyword.lower()  : 
                                 orientation = value
                                 import ast
                                 number_vector = ast.literal_eval(orientation)
                                 pixel_spacing.append(number_vector)
                                 print(cnt, 'pixel_spacing =',number_vector)
                                 
                            if 'pat'.lower() in keyword.lower()  and 'Orient'.lower() in keyword.lower()  : 
                                 orientation = value
                                 import ast
                                 number_vector = ast.literal_eval(orientation)
                                 orientations.append(number_vector)
                                 # print(cnt, 'orientation =',number_vector)
                                 orientations_slice.append(cnt-1)
                                 # break
                    
                    np.save(directory_path +'/orientations.npy', orientations) 
                    np.save(directory_path +'/pixel_spacing.npy', pixel_spacing) 
                    self.orientations=orientations
                    self.orientations_slice=orientations_slice
                    
                    print('localizer is finish' )
                    
                
                
                
    # =============================================================================
    #             ssss
    # =============================================================================
                import numpy as np
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
    
                # Load data from files
                self.orientations_slice =0,0,0
                plot_localizer=False
                try : 
                    current_slice = 5
                    orientations_all = np.load('orientations.npy') ## saed
                    pixel_spacing = np.load('pixel_spacing.npy') ## saed
                    plot_localizer=True
                except : 
                    current_slice = self.current_slice
                    orientations_all=self.orientations
                    orientations_slice = self.orientations_slice
                    
                    try : 
                        pixel_spacing=self.pixel_spacing
                    except : 1
                    
                # Create a 3D plot
                if plot_localizer  :  
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    orientations0=0
                    slice_index00=[]
                    diss=1
                    for slice_index1, orientations in enumerate(orientations_all) : 
                        slice_index = orientations_slice[slice_index1]
                        # print(not slice_index in slice_index00 , slice_index1 ,' ss ',slice_index , '==>', slice_index00)
                        if not slice_index in slice_index00  : 
                            slice_index00.append(slice_index)
                            x = range(-30000, 3000, 5000)
                            y = range(-30000, 30000, 5000)
                            if slice_index== 0 : 
                                orientations0=orientations
                            
                            
                            x, y = np.meshgrid(x, y)
                            # Assign values from loaded data
                            Xx, Xy, Xz, Yx, Yy, Yz = orientations
                            # Calculate Di and Dj based on pixel spacing
                            try : 
                                Di, Dj = pixel_spacing
                            except ValueError : 
                                Di = Dj = 1.5885
                            
                            # Define Sx, Sy, Sz (you may need to assign values to these variables)
                            Sx = Sy = Sz = 0
                            # Define M matrix
                            M = np.array([[Xx*Di, Yx*Dj, 0, Sx],
                                          [Xy*Di, Yy*Dj, 0, Sy],
                                          [Xz*Di, Yz*Dj, 0, Sz],
                                          [0, 0, 0, 1]])
                            X = np.zeros_like(x)
                            Y = np.zeros_like(y)
                            Z = np.zeros_like(x)
                            
                            for ix in range(x.shape[0]) : 
                                for iy in range(x.shape[1]) : 
                                    px = [M[0, 0] * ix, M[0, 1] * iy, 0, M[0, 3]]
                                    py = [M[1, 0] * ix, M[1, 1] * iy, 0, M[1, 3]]
                                    pz = [M[2, 0] * ix, M[2, 1] * iy, 0, M[2, 3]]
                                    pk = [M[3, 0] * ix, M[3, 1] * iy, 0, M[3, 3]]
                                    
                                    px1 = int(np.sum(px))
                                    py1 = int(np.sum(py))
                                    pz1 = int(np.sum(pz))
                                    
                                    X[ix, iy] = px1
                                    Y[ix, iy] = py1
                                    Z[ix, iy] = pz1
                            
                            
                            
                            if slice_index>  0 : 
                                # print(orientations)    
                                # print(orientations0)    
                                dif =np.sum(np.abs (np.array(orientations)-np.array(orientations0)))
                                # print(' Diff =' , dif)
                                
                                if dif==0 : 
                                    # X=X+diss
                                    # Y=Y+diss
                                    Z=Z+diss
                                    diss=diss+1
                                    
                            orientations0=orientations
                            
                            # ax.plot_surface(X, Y, Z , color='g')
                            ax.plot_surface(X, Y, Z  )
                            if current_slice == slice_index : 
                                ax.plot_surface(X, Y, Z, color='r')
                            
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title('3D Surface Plot')
                        plt.show()
    
                self.progressBar_2.setStyleSheet("QProgressBar {"
                            "border :  2px solid red;"
                            "border-radius :  5px;"
                            "background-color :  black;"
                            "color :  yellow;"  # Set text color to yellow
                            "text-align :  center;"  # Center-align text
                            "}"
                            "QProgressBar :  : chunk {"
                            "background-color :  green;"
                            "}")
        if not A:
            print('I cant find dicom files')

if __name__ == "__main__" : 
    app = QApplication(sys.argv)
    # myapp = MyForm()
    # myapp.show()
    # sys.exit(app.exec_())
    app = QApplication(sys.argv)
    main_window = MyForm()
    main_window.show()
    sys.exit(app.exec_())
    # app = QApplication(sys.argv)
    # main_window.show()
    
    # reset_filter = ResetEventFilter()
    # app.installEventFilter(reset_filter)
    
    # sys.exit(app.exec_())
    
    

        