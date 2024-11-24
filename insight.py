import os
import sys

import ast
import cv2
import pydicom
import numpy as np
import tensorflow as tf
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from localizer_draw import f_localizer_draw
from manual_segmentation_function import segmentator
from postprocessing import post_processing

global model_cardseg


def dicom_to_list(dicom_file_path) : 
    dicom_data = pydicom.dcmread(dicom_file_path)
    tag_value_list = []
    for elem in dicom_data.iterall() : 
        tag_keyword = elem.name
        tag_value = str(elem.value)
        tag_value_list.append((tag_keyword, tag_value))
    return tag_value_list

def list_dicom_files(directory_path):
    dicom_files = []
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.dcm'):
                dicom_files.append(os.path.join(dirpath, filename))
    return dicom_files

class MyForm(QMainWindow):
    def __init__(self) : 
        super().__init__()
         
        self.activateWindow()
        self.raise_()
        self.setGeometry(50, 50, 600, 400)

        self.initUI()

        self.SA_NO =0
        self.LA_NO =0
        
        self.model_cardseg_loaded= False
        
        self.all_SA_slice=[]
        self.orientations = []
        
        self.actionread_Path.triggered.connect(self.get_path_clicked)
        
        self.Preview.clicked.connect(self.Preview_function)
        
        self.actionNew_Project.triggered.connect(self.get_path_clicked)
        
        
        self.progressBar.setValue(0)
        self.progressBar_2.setValue(0)
        
        
        
        self.progressBar.setStyleSheet(
            "QProgressBar {"
            "border :  2px solid red;"
            "border-radius :  5px;"
            "background-color :  black;"
            "color :  yellow;"
            "text-align :  center;"
            "}"
            "QProgressBar :  : chunk {"
            "background-color :  blue;"
            "}"
        )

        T = [
            'Protocol',
            'dim2',
            'thick2',
            'pix2',
            'space3', 
            'Modality2',
            'info_image',
            'ID2',
            'Age2',
            'SEX2',
            'NAME2' , 
            'frame_no',
            'slice_no', 
            'SALA',
            'checkBox',
            'checkBox_2',
            'private_creator',
            'Institution_Name',
            'seriesdescription',
            'file_name',
        ]
       
        for i in T : 
            element = getattr(self, i)
            element.setStyleSheet(
                "background-color :  rgb(160, 160, 160); font :  75 12pt 'Times New Roman'; color :  rgb(0, 0, 0);"
            )
        
        self.progressBar_2.setStyleSheet(
            "QProgressBar {"
            "border :  2px solid red;"
            "border-radius :  5px;"
            "background-color :  black;"
            "color :  yellow;"
            "text-align :  center;"
            "}"
            "QProgressBar :  : chunk {"
            "background-color :  blue;"
            "}"
        )
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
    
    def initUI(self) : 
        loadUi('UI.ui', self) 
        
        self.actionCardSegNet.triggered.connect(self.on_cardsegnet_triggered)
        self.actionMECardNet.triggered.connect(self.on_mecardnet_triggered)
        # self.menuAutomatic.triggered.connect(self.)

        self.setWindowTitle(
            "  CMRI Insight :  A GUI based Open-Source Segmentation and Motion Tracking Application  ver 2.1.5"
        )
        self.get_path_button.setEnabled(True)
        
        self.get_path_button.clicked.connect(self.get_path_clicked)
        
        
        self.actionDICOM_Path.triggered.connect(self.get_path_clicked)
        
        self.actionGenerate_3D_Mesh.triggered.connect(self.actionGenerate_3D_Mesh_clicked)
        self.actionSparse_Filed.triggered.connect(self.actionSparse_Filed_clicked)
        self.actionCircumferential_Strain.triggered.connect(self.actionCircumferential_Strain_clicked)
        self.actionRadial_Strain.triggered.connect(self.actionRadial_Strain_clicked)
        
        
        self.actionGenerate_Bull_s_Eye_Plot.triggered.connect(self.actionGenerate_Bull_s_Eye_Plot_clicked)

        
        
        
        
        self.seg_all_cardsegnet.clicked.connect(self.seg_all_cardsegnet_clicked)
        self.seg_all_mecardnet.clicked.connect(self.seg_all_mecardnet_clicked)
        
        self.NF.clicked.connect(lambda :  self.SF_clicked(0, 1))
        self.PF.clicked.connect(lambda :  self.SF_clicked(0, -1))
        self.NS.clicked.connect(lambda :  self.SF_clicked(1, 0))
        self.PS.clicked.connect(lambda :  self.SF_clicked(-1, 0))
        
        
        self.creat_contour.clicked.connect(self.creat_contour_function)
        self.load_contour.clicked.connect(self.load_contour_function)
        self.Edit_contour.clicked.connect(self.Edit_contour_function)
    
    def on_cardsegnet_triggered(self):
        import numpy as np
        import cv2
        loading = cv2.imread ('loading.png')
        self.dis2(loading)
        self.dis3(loading)
        self.dis4(loading)

        from cardsegnet import CardSegNet
        def load_cardsegnet_model():
            print('self.model_cardseg_loaded',self.model_cardseg_loaded)
            if  self.model_cardseg_loaded :
                model =  model_cardseg
            if not self.model_cardseg_loaded :
                try:
                    import os
                    current_directory = os.getcwd()
                    model = CardSegNet(num_class = 3, input_shape=(128, 128,3) )()
                    print('CardSegNet is loading ...')
                    weights_directory = os.path.join(current_directory, 'weights')
                    # w_cardseg = weights_directory + '\\best_only_acdc.hdf5'
                    w_cardseg = weights_directory + '\\best_acdc_and_mm2.hdf5'
                    if os.path.exists(w_cardseg):
                        print(f"CardSegNet model loaded successfully. ({w_cardseg})")
                        model.load_weights(w_cardseg)
                        # model = tf.keras.models.load_model(w_cardseg)
                        return model
                    else:
                        print("CardSegNet model loaded with initial weights.")
                        
                        return model
                except Exception as e:
                    print(f"1 Error loading CardSegNet model: {e}")
                    return None
        
        model = load_cardsegnet_model()
        if model:
            print('CardSegNet is loaded')
            model_cardseg=model
            # self.model_cardseg_loaded = True
        # import matplotlib.pyplot as plt
        print(f"**************************{self.current_frame}")
        
        dicom_data1 = self.dicom_files [self.current_slice]
        dicom_data = pydicom.dcmread(dicom_data1)
        image = dicom_data.pixel_array
        
        
        self.image_pixel = image
        inputs = self.image_pixel[self.current_frame, :, :]
        inputs =cv2.cvtColor(inputs , cv2.COLOR_GRAY2RGB)
        inputs = inputs.astype(np.float64)
        
        
        
        image = cv2.resize(inputs, (128, 128))
        low, high = 1.00, 99.0
        low, high = np.percentile(image, (low, high))
        image[image < low] = low
        image[image > high] = high
        image = (image - low) / (high - low)
        

        plt.figure();
        plt.imshow(image)
        plt.show()

        print( image.dtype ,np.max(image) ,np.min(image) )
        output_original = model(np.expand_dims(image,0)).numpy()
        output_original = output_original[0]
        
        plt.figure();
        plt.imshow(output_original)
        plt.show()


        output_original[output_original >= 0.3] = 1
        output_original[output_original < 0.3] = 0
        output_original = output_original.astype(np.uint8)
        # _, output_postprocess= post_processing(output_original)
               
        output_postprocess = output_original ### sss
        mask_l = output_postprocess[:, :, 2]  
        mask_r = output_postprocess[:, :, 0]  
        mask_m = output_postprocess[:, :, 1]  
        
        cond1 =  np.count_nonzero(mask_l) >  0.5*mask_l.size
        cond2 =  np.count_nonzero(mask_r) >  0.5*mask_r.size
        cond3 =  np.count_nonzero(mask_m) >  0.5*mask_m.size

        print('cond1',cond1, 'cond2',cond2,'cond3',cond3,)
        if cond1 or cond2 or cond3:
            output_postprocess = output_original
            
            mask_l = output_postprocess[:, :, 2]  
            mask_r = output_postprocess[:, :, 0]  
            mask_m = output_postprocess[:, :, 1]
            
        
        
        mask_l3=0*output_postprocess; mask_l3[:, :, 0] = mask_l
        mask_r3=0*output_postprocess; mask_r3[:, :, 2] = mask_r 
        mask_m3=0*output_postprocess; mask_m3[:, :, 1] = mask_m

        print('np.shape(mask_r3 )' ,np.shape(mask_r3 ))
        
      

        result_r= image *.8+0.3*mask_r3  
        result_l= image *.8+0.3*mask_l3  
        result_m= image *.8+0.3*mask_m3 
        
        
        self.dis2(result_l)
        self.dis3(result_r)
        self.dis4(result_m)
        
        id_patient = self.patient_id
        self.path_temp = self.path+'//temp'
        try: os.mkdir(self.path_temp)
        except: pass
        mask_r = np.array(mask_r)   ;    mask_l = np.array(mask_l)  ;          mask_m = np.array(mask_m)  
        
        import cv2
        import numpy as np
        
        # Function to get points around the mask using OpenCV
        def get_points_around_mask(mask, neighborhood_size=1):
            # Ensure the mask is in binary form (0 or 255)
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
            # Apply dilation to expand the mask region by the neighborhood size
            kernel = np.ones((2 * neighborhood_size + 1, 2 * neighborhood_size + 1), np.uint8)
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
            # Subtract the original mask from the dilated mask to get the points around the mask
            points_around_mask = dilated_mask - binary_mask
        
            # Get the coordinates of the non-zero points around the mask
            points = np.column_stack(np.where(points_around_mask > 0))
        
            return points
        
        
        mask_r = np.array(mask_r)
        mask_l = np.array(mask_l)
        mask_m = np.array(mask_m)
        
        neighborhood_size = 1
        
        # Get points around each mask
        point_r = get_points_around_mask(mask_r, neighborhood_size)
        point_l = get_points_around_mask(mask_l, neighborhood_size)
        
        # For point_m, combine mask_r and mask_m, then get the points around the combined mask
        combined_mask = cv2.add(mask_r, mask_m)
        point_m = get_points_around_mask(combined_mask, neighborhood_size)
        
        # Now point_r, point_l, and point_m contain the points around the masks
        
        point_r = np.array(point_r) ;    point_l = np.array(point_l);          point_m = np.array(point_m)   
        saved_image = np.array(image)     
        p1=self.path_temp+'// patient' +str(id_patient)+'frame'+str(self.current_frame)+'slice'+str(self.current_slice)+'.npz'
                                                                      
        np.savez(p1, saved_image=saved_image, mask_r=mask_r,mask_m=mask_m,mask_l=mask_l,point_r=point_r,point_l=point_l,point_m=point_m,)
           
        
        
            
    def on_mecardnet_triggered(self):
        def load_mecardnet_model():
            try:
                import os
                current_directory = os.getcwd()
                # print('mecardnet is loading ...')
                weights_directory = os.path.join(current_directory, 'weights')
                w_cardseg= weights_directory+ '\MECardNet.hdf5'
                if os.path.exists(w_cardseg):
                    model = tf.keras.models.load_model(w_cardseg)
                    print("mecardnet model loaded successfully.")
                    return model
            except Exception as e:
                print(f"2 Error loading mecardnet model: {e}")
                return None
        
        model = load_mecardnet_model()
        if model:
            print('mecardnet is loaded')
    
    def seg_all_cardsegnet_clicked(self) : 
        print(':)')
        pass
        
    def seg_all_mecardnet_clicked(self) : 
        print(':)')
        pass
        
    def actionGenerate_3D_Mesh_clicked(self) : 
        
        
        print(':)')
        pass
    
    
    def actionSparse_Filed_clicked(self) : 
        print(':)')
        pass
    
    
    def actionCircumferential_Strain_clicked(self) : 
        localizer_image = cv2.imread ('E:\0001phd\00_thesis\0_mfiles\1_DATA_Rajaii\all_strains\strain285859\JAHANDAR_ALI-19351029-2016-10-29-Polarmap-Peak_Circumferential_Displacement_(deg).png')
        self.dis5(localizer_image)
        print(':)')
        pass
    
    
    def actionRadial_Strain_clicked(self) : 
        localizer_image = cv2.imread ('E:\0001phd\00_thesis\0_mfiles\1_DATA_Rajaii\all_strains\strain285859\JAHANDAR_ALI-19351029-2016-10-29-Polarmap-Peak_Radial_Displacement_(mm).png')
        self.dis5(localizer_image)
        print(':)')
        pass
    # ssssss
    def actionGenerate_Bull_s_Eye_Plot_clicked(self) :
        localizer_image = cv2.imread ('E:\0001phd\00_thesis\0_mfiles\1_DATA_Rajaii\all_strains\strain285859\JAHANDAR_ALI-19351029-2016-10-29-Polarmap-Peak_Radial_Displacement_(mm).png')
        self.dis5(localizer_image)
        
        print(':)')
        pass
    
    def sss(self) : 
        print(':)')
        pass    
      
    def sss(self) : 
        print(':)')
        pass    
      
    def sss(self) : 
        print(':)')
        pass    
      
    def sss(self) : 
        print(':)')
        pass    
    
    
    
    
    
    
    def get_path_clicked(self) : 
        
        self.NF.setEnabled(True)
        self.PF.setEnabled(True)
        self.NS.setEnabled(True)
        self.PS.setEnabled(True)
        
        self.current_slice =0
        self.current_frame = 0
        # Change the background color
        self.seg_all_cardsegnet.setStyleSheet("background-color: rgb(150, 250, 150);")
        self.seg_all_mecardnet.setStyleSheet("background-color: rgb(150, 250, 150);")
        
        # Simulate activating (clicking) the buttons
        self.seg_all_cardsegnet.click()  # This will simulate a click on seg_all_cardsegnet
        self.seg_all_mecardnet.click()  # This will simulate a click on seg_all_mecardnet

        # If you want to keep the buttons checked (if they are checkable)
        self.seg_all_cardsegnet.setChecked(True)
        self.seg_all_mecardnet.setChecked(True)

        
        self.creat_contour.setEnabled(True)
        self.load_contour.setEnabled(True)
        
        self.creat_contour.setStyleSheet("background-color :  rgb(150,250, 150);")
        self.load_contour.setStyleSheet("background-color :  rgb(150,250, 150);")
        self.Edit_contour.setStyleSheet("background-color :  rgb(150,250, 150);")
        
        
        
        directory_path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        
        self.path = str(directory_path)
        
        try:
            split_parts = self.path.split('viewer/')
        
            after_viewer = split_parts[1]
        except:
            after_viewer = self.path
        
        dicom_files = list_dicom_files(directory_path)
        A = len(dicom_files) > 0  

        self.path= str(directory_path)
        try:
            split_parts = self.path.split('viewer/')
    
            after_viewer = split_parts[1]
        except:
            after_viewer = self.path
        
        if A:
            if  not 'DICOM' in after_viewer or not 'dicom' in after_viewer : 
                self.path=self.path+'//DICOM'
            
            self.directory_path = directory_path
            
            if directory_path : 
                dicom_files = list_dicom_files(directory_path)
                self.dicom_files=dicom_files

                dicom_data = pydicom.dcmread(dicom_files[0])
                patient_id = dicom_data.get("PatientID", "N/A")
                self.patient_id=patient_id
                patient_id = dicom_data.PatientID
                patient_name = dicom_data.PatientName 
                
                SA_NO = 0
                LA_NO = 0                  
    
                for dicom_data1 in dicom_files : 
                    dicom_data = pydicom.dcmread(dicom_data1)
                    image = dicom_data.pixel_array
                    
                    if np.size(np.shape(image))>2 : 
                        frame_no001=np.shape(image)[0]
                        dicom_dict = []
                        
                        for data_element in dicom_data : 
                            tag = data_element.tag
                            value = data_element.value
                            dicom_dict.append({'tag' :  tag, 'value' :  value})
                        
                        dicom_file_path = dicom_data1
                        tags_and_values = dicom_to_list(dicom_file_path)
                        
                        private_creator = None
                        orientation = None
                        inst11 = None
                        seriesdescription=None
                        sex=None
                        modality=' MR'
                        manufacturer=None
                        pat_name=None
                        pat_ID=None
                        code_meaning=None
                        pat_size=None
                        pat_weight=None
                        pat_b_d=None
                        address=None
                        date_pat=None
                        date_study=None
                        Protocol=None
                        tthick2=None
                        ppix2=None
                        sspace3=None 
                        
                        for tag, value in tags_and_values : 
                            keyword=tag
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
                        break
                
                dicom_data1 = self.dicom_files [self.current_slice]
                
                dicom_data = pydicom.dcmread(dicom_data1)
                
                image = dicom_data.pixel_array
                self.image_pixel = image
                image0=image
                
                localizer_image = cv2.imread ('localizer.JPG')
                self.dis5(localizer_image)
                
                T = [
                    'Protocol',
                    'dim2',
                    'thick2',
                    'pix2',
                    'space3',
                    'Modality2',
                    'info_image',
                    'ID2',
                    'Age2',
                    'SEX2',
                    'NAME2', 
                    'frame_no',
                    'slice_no', 
                    'SALA',
                    'checkBox',
                    'checkBox_2',
                    'private_creator',
                    'Institution_Name',
                    'seriesdescription',
                    'file_name'
                ]
               
                for i in T : 
                    element = getattr(self, i)
                    element.setStyleSheet(
                        "background-color :  rgb(160, 160, 240); font :  75 12pt 'Times New Roman'; color :  rgb(0, 0, 0);"
                    )
                
                T2=[
                    'PS_3',
                    'PS_4',
                    'PS_5',
                    'PS_6',
                ]
                
                for i in T2 :
                    element = getattr(self, i)
                    element.setStyleSheet(
                        "background-color :  rgb(255, 160, 160); font :  75 12pt 'Times New Roman'; color :  rgb(0, 0, 0);"
                    )
                
                    
                  
                
                patient_name=pat_name
                try : 
                    parts = patient_name.split('^'); patient_name = parts[1] + ' ' + parts[0]
                except : 1
    
                self.ID2.setText('ID :  '+ str(patient_id))
                self.NAME2.setText('NAME :  '+ str(patient_name)  )
                self.slice_no.setText('Slice No :  '+ str(len(dicom_files)))
                self.frame_no.setText('Frame No :  '+ str(frame_no001))
                self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))
                self.data_Orientation.setText('Orientation :  '+str(orientation))
                self.private_creator.setText('Private Creator : '+str(private_creator))
                self.Modality2.setText(' Modality : '+str(modality))
                address2=''
                
                for i in address:
                    if i.isalpha():address2=address2 + i
                    else:address2=address2 +  ' '
    
                self.Institution_Name.setText('Institution Name :  '+str(inst11)+' , '+ str (address2))
                self.seriesdescription.setText('Series Description :  '+str(seriesdescription))
                
                try:
                    age=str (round (  int(date_study)/10000 - int(date_pat)/10000)  )
                except: 
                    age=' 53'
                
                dimm=str(np.shape(self.image_pixel))
                
                Protocol ='T1WI'
                
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
                cv2.waitKey(10)
                
                cnt=0
                
                self.checkBox.stateChanged.connect(self.checkbox_state_changed)
                self.checkBox_2.stateChanged.connect(self.checkbox_state_changed2)
                
                if self.checkBox1:
                    self.dicom_files= dicom_files
                    for dicom_data1 in dicom_files : 
                        cnt=cnt+1
                        a=int (np. round (100*cnt/len(dicom_files)))
                        self.progressBar.setValue(a)
                        
                        self.progressBar.setStyleSheet(
                            "QProgressBar {"
                            "border :  2px solid red;"
                            "border-radius :  5px;"
                            "background-color :  black;"
                            "color :  yellow;"
                            "text-align :  center;"
                            "}"
                            "QProgressBar :  : chunk {"
                            "background-color :  blue;"
                            "}"
                        )
                        
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

                self.progressBar.setStyleSheet(
                    "QProgressBar {"
                    "border :  2px solid red;"
                    "border-radius :  5px;"
                    "background-color :  black;"
                    "color :  yellow;"
                    "text-align :  center;"
                    "}"
                    "QProgressBar :  : chunk {"
                    "background-color :  green;"
                    "}"
                )

                cv2.waitKey(100)

                orientations=[]

                pixel_spacing=[]

                cnt=0

                orientations_slice=[]

                if self.checkBox2: 
                    for dicom_data1 in dicom_files : 
                        cnt=cnt+1
                        
                        a = int(np.round(100*cnt/len(dicom_files)))
                        
                        self.progressBar_2.setValue(a)
                        
                        self.progressBar_2.setStyleSheet(
                            "QProgressBar {"
                            "border :  2px solid red;"
                            "border-radius :  5px;"
                            "background-color :  black;"
                            "color :  yellow;"
                            "text-align :  center;"
                            "}"
                            "QProgressBar :  : chunk {"
                            "background-color :  blue;"
                            "}"
                        )

                        dicom_file_path = dicom_data1

                        tags_and_values = dicom_to_list(dicom_file_path)

                        cc=0
                        for tag, value in tags_and_values : 
                            keyword=tag
                            
                            cc=cc+1
                            
                            if 'pixel'.lower() in keyword.lower()  and 'space'.lower() in keyword.lower()  : 
                                orientation = value
                                
                                number_vector = ast.literal_eval(orientation)
                                pixel_spacing.append(number_vector)
                                 
                            if 'pat'.lower() in keyword.lower()  and 'Orient'.lower() in keyword.lower()  : 
                                orientation = value

                                number_vector = ast.literal_eval(orientation)
                                orientations.append(number_vector)
                                orientations_slice.append(cnt-1)
                    
                    np.save(directory_path +'/orientations.npy', orientations) 
                    np.save(directory_path +'/pixel_spacing.npy', pixel_spacing) 
                    self.orientations=orientations
                    self.orientations_slice=orientations_slice
                
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
                    
                    try: 
                        pixel_spacing=self.pixel_spacing
                    except: 1
                
                if plot_localizer  :  
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    orientations0=0
                    slice_index00=[]
                    diss=1
                    for slice_index1, orientations in enumerate(orientations_all) : 
                        slice_index = orientations_slice[slice_index1]
                        
                        if not slice_index in slice_index00  : 
                            slice_index00.append(slice_index)
                            x = range(-30000, 3000, 5000)
                            y = range(-30000, 30000, 5000)
                            if slice_index== 0 : 
                                orientations0=orientations
                            
                            
                            x, y = np.meshgrid(x, y)
                            Xx, Xy, Xz, Yx, Yy, Yz = orientations
                            
                            try : 
                                Di, Dj = pixel_spacing
                            except ValueError : 
                                Di = Dj = 1.5885
                            
                            Sx = Sy = Sz = 0
                            
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
                                dif =np.sum(np.abs (np.array(orientations)-np.array(orientations0)))
                                
                                if dif==0 : 
                                    Z=Z+diss
                                    diss=diss+1
                                    
                            orientations0=orientations

                            ax.plot_surface(X, Y, Z  )
                            if current_slice == slice_index : 
                                ax.plot_surface(X, Y, Z, color='r')
                            
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title('3D Surface Plot')
                        plt.show()
    
                self.progressBar_2.setStyleSheet(
                    "QProgressBar {"
                    "border :  2px solid red;"
                    "border-radius :  5px;"
                    "background-color :  black;"
                    "color :  yellow;"
                    "text-align :  center;"
                    "}"
                    "QProgressBar :  : chunk {"
                    "background-color :  green;"
                    "}"
                )
        if not A:
            print('I cant find dicom files')

    def creat_contour_function(self) :
        
        dicom_data1 = self.dicom_files [self.current_slice]

        dicom_data = pydicom.dcmread(dicom_data1)
        image = dicom_data.pixel_array
        self.image_pixel = image
        currnet_image= self.image_pixel [self.current_frame, : , :  ]
        
        try: 
            if np.size(np.shape(currnet_image))>2 :  currnet_image=currnet_image[self.current_frame ,  : , : ]
            self.currnet_image= currnet_image 
        except: 1
        
        i = self.current_frame
        
        try : 
            saved_image, mask_r  ,mask_l , mask_m ,point_r, point_l,  point_m = segmentator(currnet_image,i)
            
            try : 
                rgb_image = cv2.cvtColor(saved_image, cv2.COLOR_GRAY2RGB)
            except : 
                rgb_image = saved_image
            
            mask_l3 = 0*rgb_image;mask_l3[ : , : ,0] = mask_l
            mask_r3 = 0*rgb_image;mask_r3[ : , : ,2] = mask_r
            mask_m3 = 0*rgb_image;mask_m3[ : , : ,1] = mask_m

            result_r = cv2.addWeighted(rgb_image, 1, mask_r3, 0.3, 0)
            result_l = cv2.addWeighted(rgb_image, 1, mask_l3, 0.3, 0)
            result_m = cv2.addWeighted(rgb_image, 1, mask_m3, 0.3, 0)
            color1 = (0, 0, 255)
            color3 = (255, 0, 0)
            color2 = (0, 255, 255)
            
            for x,y in point_r  : 
                result_r = cv2.circle(result_r, (x, y), 1, color1, 1)
            for x,y in point_l  : 
                result_l= cv2.circle(result_l, (x, y), 1, color2, 1)
            for x,y in point_m  : 
                result_m = cv2.circle(result_m, (x, y), 1, color3, 1)
            
            self.dis3(result_r)
            self.dis2(result_l)
            self.dis4(result_m)
            
            
            
            id_patient = self.patient_id
            self.path_temp = self.path+'//temp'
            try: os.mkdir(self.path_temp)
            except: pass
            mask_r = np.array(mask_r)   ;    mask_l = np.array(mask_l)  ;          mask_m = np.array(mask_m)  
            point_r = np.array(point_r) ;    point_l = np.array(point_l);          point_m = np.array(point_m)   
            saved_image = np.array(saved_image)     
            p1=self.path_temp+'// patient' +str(id_patient)+'frame'+str(self.current_frame)+'slice'+str(self.current_slice)+'.npz'
                                                                          
            np.savez(p1, saved_image=saved_image, mask_r=mask_r,mask_m=mask_m,mask_l=mask_l,point_r=point_r,point_l=point_l,point_m=point_m,)
        except:
            pass

    def load_contour_function(self) :
        self.loading=False
        try : 
            self.path_temp = self.path+'//temp'

            if not os.path.exists(self.path_temp) : 
                os.mkdir(self.path_temp)
            
            id_patient = self.patient_id
            p1=self.path_temp+'// patient' +str(id_patient)+'frame'+str(self.current_frame)+'slice'+str(self.current_slice)+'.npz'
            
            
            print(10*'s',p1, os.path.exists(p1))
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

                point_r = loaded_arrays['point_r']
                point_l = loaded_arrays['point_l']
                point_m = loaded_arrays['point_m']
                saved_image= loaded_arrays['saved_image']
                rgb_image = saved_image
                
                mask_l3 = 0*rgb_image;mask_l3[ : , : ,0] = mask_l
                mask_r3 = 0*rgb_image;mask_r3[ : , : ,2] = mask_r
                mask_m3 = 0*rgb_image;mask_m3[ : , : ,1] = mask_m
                result_r = cv2.addWeighted(rgb_image, 1, mask_r3, 0.3, 0)
                result_l = cv2.addWeighted(rgb_image, 1, mask_l3, 0.3, 0)
                result_m = cv2.addWeighted(rgb_image, 1, mask_m3, 0.3, 0)
                color1 = (0, 0, 255)
                color3 = (255, 0, 0)
                color2 = (0, 255, 255)
                
                for x,y in point_r  : 
                    result_r = cv2.circle(result_r, (x, y), 1, color1, 1)
                for x,y in point_l  : 
                    result_l= cv2.circle(result_l, (x, y), 1, color2, 1)
                for x,y in point_m  : 
                    result_m = cv2.circle(result_m, (x, y), 1, color3, 1)
               
                self.dis3(result_r)
                self.dis2(result_l)
                self.dis4(result_m)
                
                self.loading = True
        except : 
            self.loading = False

    def Edit_contour_function(self) : 
        print('Edit_contour_function is not completed')

    def Preview_function(self) : 
        all_SA_slice=[]

        cnt=0
        LA_NO = 0
        SA_NO = 0

        if self.checkBox1: 
            dicom_files=self.dicom_files
            
            for dicom_data1 in dicom_files : 
                cnt = cnt+1
                a = int(np.round(100*cnt/len(dicom_files)))
                
                self.progressBar.setValue(a)
                
                self.progressBar.setStyleSheet(
                    "QProgressBar {"
                    "border :  2px solid red;"
                    "border-radius :  5px;"
                    "background-color :  black;"
                    "color :  yellow;"
                    "text-align :  center;"
                    "}"
                    "QProgressBar :  : chunk {"
                    "background-color :  blue;"
                    "}"
                )
                
                dicom_file_path = dicom_data1
                tags_and_values = dicom_to_list(dicom_file_path)
                
                for tag, value in tags_and_values : 
                    if 'series'.lower() in tag.lower()  and 'description'.lower() in tag.lower()  : 
                        seriesdescription = value
                        t = 0
                        
                        if 'LA'.lower()in value.lower() or '2ch'in value.lower() or '3ch'.lower()in value.lower() or '4ch'in value.lower() : 
                            LA_NO = LA_NO + 1
                            t = 1
                            self.LA_NO =LA_NO
                        
                        if t==0 and 'Short'.lower()in value.lower() or 'SA'.lower()in value.lower() : 
                            SA_NO = SA_NO + 1
                            t=1
                            all_SA_slice.append (cnt -1)
                            self.SA_NO = SA_NO
        
        self.all_SA_slice=all_SA_slice
        self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))                 
        self.NS_2.setText( str(self.current_slice+1) )
        self.NF_2.setText( str(self.current_frame+1) )
         
        self.progressBar.setStyleSheet(
            "QProgressBar {"
            "border :  2px solid red;"
            "border-radius :  5px;"
            "background-color :  black;"
            "color :  yellow;"
            "text-align :  center;"
            "}"
            "QProgressBar :  : chunk {"
            "background-color :  green;"
            "}"
        )
        
        cv2.waitKey(100)

        orientations=[]
        pixel_spacing=[]
        cnt=0
        orientations_slice=[]
        
        try : 
            dicom_files = self.dicom_files
            if self.checkBox2: 
                for dicom_data1 in dicom_files : 
                    cnt = cnt + 1
                    a = int(np.round(100*cnt/len(dicom_files)))
                    
                    self.progressBar_2.setValue(a)
                    
                    self.progressBar_2.setStyleSheet(
                        "QProgressBar {"
                        "border :  2px solid red;"
                        "border-radius :  5px;"
                        "background-color :  black;"
                        "color :  yellow;"
                        "text-align :  center;"
                        "}"
                        "QProgressBar :  : chunk {"
                        "background-color :  blue;"
                        "}"
                    )
                    
                    dicom_file_path = dicom_data1

                    tags_and_values = dicom_to_list(dicom_file_path)

                    cc=0
                    for tag, value in tags_and_values : 
                        keyword = tag
                        cc = cc + 1
                        if 'pixel'.lower() in keyword.lower()  and 'space'.lower() in keyword.lower()  : 
                             orientation = value
                             number_vector = ast.literal_eval(orientation)
                             pixel_spacing.append(number_vector)
                             
                        if 'pat'.lower() in keyword.lower()  and 'Orient'.lower() in keyword.lower()  : 
                             orientation = value
                             number_vector = ast.literal_eval(orientation)
                             orientations.append(number_vector)
                             orientations_slice.append(cnt-1)
                
                self.orientations=orientations
                self.orientations_slice=orientations_slice
                
                self.progressBar_2.setStyleSheet(
                    "QProgressBar {"
                    "border :  2px solid red;"
                    "border-radius :  5px;"
                    "background-color :  black;"
                    "color :  yellow;"
                    "text-align :  center;"
                    "}"
                    "QProgressBar :  : chunk {"
                    "background-color :  green;"
                    "}"
                )
                
                if 'sa' in self.seriesdescription1.lower() and self.checkBox and self.checkBox2:
                    current_slice=self.current_slice
                    for i in range (len (self.all_SA_slice)) : 
                        if current_slice ==self.all_SA_slice [i] : 
                            current_slice_sa = i
                            
                    all_SA_slices = len (self.all_SA_slice)
                    image_localizer = f_localizer_draw (current_slice_sa ,  all_SA_slices)
                    self.dis5( image_localizer)
        except:
            pass
    
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
        
        
        self.P_ED.setStyleSheet("background-color :  rgb(100,100, 100);")
        self.P_ES.setStyleSheet("background-color :  rgb(100,100, 100);")
        self.P_SA.setStyleSheet("background-color :  rgb(100,100, 100);")
        self.P_LA.setStyleSheet("background-color :  rgb(100,100, 100);")
        if self.current_frame == 0:
            self.P_ED.setStyleSheet("background-color :  rgb(150,250, 150);")
        if self.current_frame == 13:
            self.P_ES.setStyleSheet("background-color :  rgb(150,250, 150);")   
            
        temp=self.seriesdescription1.lower()
        if 'sa' in temp:
            self.P_SA.setStyleSheet("background-color :  rgb(150,250, 150);")
        if '2ch' in temp or '3ch' in temp or '4ch' in temp :
            self.P_LA.setStyleSheet("background-color :  rgb(150,250, 150);") 
        if not 'sa' in temp or '2ch' in temp or '3ch' in temp or '4ch' in temp:
            self.P_LA.setStyleSheet("background-color :  rgb(150,200, 200);")
            self.P_SA.setStyleSheet("background-color :  rgb(150,200, 200);")
            
            
            

    def dis2(self, image0) : 
        # self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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
        # self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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
        # self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 430
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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
        # self.currnet_image = image0
        # Resize the image while preserving aspect ratio
        aspect_ratio = image0.shape[1] / image0.shape[0]
        new_height = 320
        new_width = int(new_height * aspect_ratio)
        resized = cv2.resize(image0, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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

    def checkbox_state_changed(self, state) : 
        if state == 2:
            self.checkBox1=True
        else : 
            self.checkBox1=False
    
    def checkbox_state_changed2(self, state) : 
        if state == 2 :
            self.checkBox2=True
        else : 
            self.checkBox2=False

    def SF_clicked(self,S,F ) :
           
            
            
        self.load_contour_function()
                
        if not self.loading  :  
            image0=np.zeros([100,100])
            self.dis2( image0);self.dis3( image0);self.dis4( image0)
        
        if F==1 : 
            self.NF.setStyleSheet("background-color :  red;")
            if self.current_frame <self.all_frames-1 : 
                self.current_frame=self.current_frame +1  
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
        
        self.load_contour_function()
        dicom_data1 = self.dicom_files [self.current_slice]

        dicom_data = pydicom.dcmread(dicom_data1)
        image = dicom_data.pixel_array
        self.image = np.uint8(image[self.current_frame , : , : ])
        image0=image
        
        try: 
            if np.size(np.shape(image))>2 : 
                image0=image[self.current_frame ,  : , : ]
                currnet_image=image0
            self.currnet_image= currnet_image
        except: 1
        
        self.dis1(image0) 
        
        dicom_data = pydicom.dcmread(self.dicom_files [self.current_slice])
        patient_id = dicom_data.get("PatientID", "N/A")
        
        patient_id = dicom_data.PatientID
        patient_name = dicom_data.PatientName 
        
        SA_NO=0
        LA_NO=0
        
        dicom_file_path=self.dicom_files [self.current_slice]
 
        tags_and_values = dicom_to_list(dicom_file_path)
        private_creator = None
        orientation = None
        inst11 = None
        seriesdescription=None
        sex=None
        modality='MRI'
        manufacturer=None
        pat_name=None
        pat_ID=None
        code_meaning=None
        pat_size=None
        pat_weight=None
        pat_b_d=None
        address=None
        date_pat=None
        date_study=None
        Protocol=None
        tthick2=None
        ppix2=None
        sspace3=None 
        
        for tag, value in tags_and_values : 
            keyword = tag
            
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
                
        patient_name=pat_name
        
        try : 
            parts = patient_name.split('^'); patient_name = parts[1] + ' ' + parts[0]
        except : 1
        
        try:
            age=str (round (  int(date_study)/10000 - int(date_pat)/10000)  )
        except:
            age=' 53'
        
        dimm = str(np.shape(self.image_pixel))

        Protocol ='T1WI'

        self.ID2.setText('ID :  '+ str(patient_id))
        self.NAME2.setText('NAME :  '+ str(patient_name))
        self.SEX2.setText('SEX : '+sex)
        self.Age2.setText('Age : '+age)
        
        self.slice_no.setText('Slice No :  '+ str(self.all_slices))
        self.frame_no.setText('Frame No :  '+ str(self.all_frames))
        self.SALA.setText('SA = '+str(self.SA_NO)+' , LA =  '+ str(self.LA_NO))
        self.data_Orientation.setText('Orientation :  '+str(orientation))
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
        filename = os.path.basename(self.dicom_files [self.current_slice])
        
        self.file_name.setText('File name :  '+filename)
        
        image0=image
        
        if np.size(np.shape(image))>2 : 
            image0=image[self.current_frame ,  : , : ]
        
        self.dis1(image0)
        
        self.load_contour_function()
        
        if 'sa' in self.seriesdescription1.lower()  and   np.abs (S)>0.5 and self.checkBox and self.checkBox2  : 
            from localizer_draw import f_localizer_draw
            current_slice=self.current_slice
            for i in range (len (self.all_SA_slice)) : 
                if current_slice ==self.all_SA_slice [i] : 
                    current_slice_sa = i
                    
            all_SA_slices = len (self.all_SA_slice)
            image_localizer = f_localizer_draw (current_slice_sa ,  all_SA_slices)
            self.dis5( image_localizer)
        
        cv2.waitKey(10)
        cnt=0


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    main_window = MyForm()
    main_window.show()
    sys.exit(app.exec_())