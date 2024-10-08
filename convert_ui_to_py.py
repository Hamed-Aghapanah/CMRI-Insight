"""   Created on Tue May 16 12:12:06 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

# """
# from subprocess import call
# call(["python", "convert_ui_to_py.py"])

# import subprocess
# # proc = subprocess.Popen('cmd.exe', stdin = subprocess.PIPE, stdout = subprocess.PIPE)
# proc.stdin.write("cd Documents")
# import os 
# os.startfile('cmd.exe')
# proc.stdin.write("cd Documents")
import os
import keyboard
import time

os.system("start cmd")

# Because we don't want the `keyboard` module to write before cmd gets opened.
time.sleep(0.5)

keyboard.write("pyuic5 G3.ui -o G3.py")
keyboard.press_and_release("enter")
keyboard.write("exit")

keyboard.press_and_release("enter")
# os.close()