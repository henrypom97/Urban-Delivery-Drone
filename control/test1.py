import sys                                    
import traceback
from tello import DJItellopy
import av
import cv2.cv2 as cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
import datetime
import scipy.stats as sstats
import logging

drone = DJItellopy.Tello()

drone.connect()

drone.streamon()
frame_read = drone.get_frame_read()


print(drone.get_battery())

def OpenCV():
            
    frame_skip = 300                               #動画接続前
        
    while True:
      if 0 < frame_skip: #フレームスキップ処理
        frame_skip = frame_skip - 1
        continue
          
      start_time = time.time()                   # time.time UNIX time(0:0:0)からの経過時間

      img = frame_read.frame
        
      image_origin = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)     #RGB convert
      
      cv2.imshow('image_origin',image_origin)

def main():  


    time.sleep(5)

    while True:   

      print(drone.get_distance_tof())

if __name__ == '__main__' :
  main()