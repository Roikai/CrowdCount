from head_detection import detect
import os
import glob as gb
import time
import numpy as np



path=r"G:\picture2\*.png"
img=gb.glob(path)
for i in img:
    num = detect(i, './checkpoints/head_detector_final')
