"""   Created on Fri Jan 19 20:28:01 2024

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer, QTime, Qt

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QPushButton
        self.button = QPushButton("Click me to update time")

        # Connect the button to the update_time function
        self.button.clicked.connect(self.update_time)

        # Create a timer to update the time every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every 1000 milliseconds (1 second)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)

        # Set initial time
        self.update_time()

    def update_time(self):
        # Get the current time
        current_time = QTime.currentTime()

        # Format the time as a string (hh:mm:ss)
        time_str = current_time.toString(Qt.DefaultLocaleLongDate)

        # Update the button text
        self.button.setText(time_str)

if __name__ == '__main__':
    app = QApplication([])
    window = MyWidget()
    window.show()
    app.exec_()
