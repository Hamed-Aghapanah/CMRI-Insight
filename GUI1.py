import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QPushButton
from PyQt5.QtWidgets import QDialog, QApplication, QGraphicsScene, QGraphicsPixmapItem, QColorDialog,QFontDialog
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog

from PyQt5 import uic
import datetime
from time import gmtime, strftime
import sys
import cv2
from PyQt5 import QtWidgets, QtGui
import numpy as np
from time import gmtime, strftime
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor

# from GraphicsView import *


class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.datee=0
        import cv2
        # fig_logo
        image0=cv2.imread('imagess/1.png')
        self.ui.fig_logo(image0)
        self.ui.Author.clicked.connect(self.Author_function)
        
        
        

# def dialog():
#     mbox = QMessageBox()

#     mbox.setText("Your allegiance has been noted")
#     mbox.setDetailedText("You are now a disciple and subject of the all-knowing Guru")
#     mbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            
#     mbox.exec_()

if __name__ == "__main__":
    MyForm, Window = uic.loadUiType("G2.ui")
    app = QApplication([])
    Window1 = Window()
    MyForm1 = MyForm()
    
    # myapp = MyForm()
    # myapp.show()
    # sys.exit(app.exec_())
    
    
    MyForm1.setupUi(Window1)
    Window1.show()
    print(1)
    class MyForm(QMainWindow):
        def __init__(self):
            super().__init__()
            self.datee=0
            import cv2
            # fig_logo
            image0=cv2.imread('imagess/1.png')
            self.ui.fig_logo(image0)
    app.exec()
    print(2)
print(3)    
    # self.fig_logo(image0)
    
    
    # app = QApplication(sys.argv)
    # myapp = MyForm()
    # myapp.show()
    # sys.exit(app.exec_())