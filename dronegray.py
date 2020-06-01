import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
import datetime
import math

def main():
  drone = tellopy.Tello()
   
  try:
    drone.connect()
    drone.wait_for_connection(60.0)
    
    retry = 3
    container = None
    while container is None and 0 < retry:
      retry -= 1
      try:
        # 動画の受信開始を処理
        container = av.open(drone.get_video_stream())
      except av.AVError as ave:
        print(ave)
        print('retry...')
          
    frame_skip = 300
    while True:
      for frame in container.decode(video=0):
        if 0 < frame_skip: #フレームスキップ処理
          frame_skip = frame_skip - 1
          continue
        start_time = time.time()
        #img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_mask = cv2.GaussianBlur(gray, (15, 15), 0)
        canny = cv2.Canny(img_mask, 100, 150)
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(canny)
        ax.axis("off")

        #avg = None
        #if avg is None:
        #      avg = gray.copy().astype("float")
        #      continue
        #cv2.accumulateWeighted(gray,avg,0.5)
        #flameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        thresh = cv2.threshold(canny, 0, 256, cv2.THRESH_BINARY)[1] #黒を排除する必要あり

        #輪郭の抽出
        #輪郭の階層情報
        cnts, contours, img  = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('img', img) 
        cv2.imshow('image', image)
        cv2.waitKey(1)
        
        
        if frame.time_base < 1.0/60:
            time_base = 1.0/60
        else:
            time_base = frame.time_base
        # フレームスキップ値を算出
        frame_skip = int((time.time() - start_time) / time_base)

  except Exception as ex:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print(ex)
  finally:
    drone.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
  main()
