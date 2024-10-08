"""   Created on Sun May 14 00:08:13 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
# Form1, Window = uic.loadUiType("G1.ui")
import sys
from PyQt5 .QtWidgets import QDialog 
from     G1 import *
class Form (QDialog):
    def __init__(self):
        s=10;print(s)
        
        super().__int__()
        s=11;print(s)
        self.ui.setupUi(self)
        # self.ui.ButtonClickMe.clicked.connect(self.dispmessage)
        import cv2
        image0=cv2.imread('images/Slide1.PNG')
        s=12;print(s)
        self.dis1(image0)

        s=13;print(s)
        self.show()
        s=14;print(s)
        

        
    def dispmessage (self):
        self.ui.labelResponse.setText("Hello"+self.ui.lineEditName.text())
   
        
if __name__=="__main__":
    s=0;print(s)
    app= QApplication (sys.argv)
    w= Form()
    w.show()
    sys.exit(app.exec)
    
    
    # app = QApplication([])
    # s=1;print(s)
    # window = Window()
    # s=2;print(s)
    # form = Form()
    # s=3;print(s)
    # form.setupUi(window)
    # s=4;print(s)
    # window.show()
    # s=5;print(s)
    # app.exec()
    # s=6;print(s)
    # sys.exit(app.exec)