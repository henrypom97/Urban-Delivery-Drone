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
        
        image_origin = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        
        img_mask = cv2.GaussianBlur(image_origin, (15, 15), 0)  #フィルタの中の引数
        canny = cv2.Canny(img_mask, 100, 150)

        #色の抽出
        
        def red_detect(canny):
        
          RGBLower1 = np.array([62, 60, 100])    # 抽出する色の下限(BGR)
          RGBUpper1 = np.array([75 ,80, 137])   # 抽出する色の上限(BGR)
          mask1 = cv2.inRange(img_mask, RGBLower1, RGBUpper1) 

          #RGBLower2 = np.array([202, 185, 115])
          #RGBUpper2 = np.array([255, 255, 148])
          #mask2 = cv2.inRange(image_origin,RGBLower2, RGBUpper2)

          return mask1 

        mask = red_detect(canny)

        #img_mask = cv2.GaussianBlur(mask, (15, 15), 0)  #フィルタの中の引数
        #canny = cv2.Canny(img_mask, 200, 300)

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(mask)
        ax.axis("off")


        #輪郭の抽出
        #輪郭の階層情報
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        i = 0 
        for contour in contours: 
          
          #for i in range(6):

          cv2.drawContours(image_origin, contours, i, (255,0,0), 5)

          S = cv2.contourArea(contours[i])

          if S>0:
            print(S)
          
          i += 1
      
        #for counter in counters:
              #x, y, w, h ~ cv2.boundingRect(contour)
              #cv2.rectangle(image_origin, (x,y), (x+w, y+h), (0,255,0),3)
              
              #print(x)
              #print(y)
              #priny(w)
              #print(h)   

  # Harrisコーナ検出

        #mask = np.uint16(mask)
        #gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        #mask = np.float32(mask)

        #dst = cv2.cornerHarris(mask, 2, 9, 0.16)

        # 青の点を頂点に打つ
        #mask[dst>0.01*dst.max()] = [255,0,0]

        #青のpixelを探す
        #coord = np.where(np.all(mask == (255, 0, 0), axis = -1))

        #printt coorsdinate
        #for i in range(len(coord[0])):
              #print("X:%s Y:%s"%(coord[1][i],coord[0][i]))

      #for countour in contours:
        #x, y, w, h = cv2.boundingRect(contour)
       # cv2.rectangle(image_origin, (x,y), (x+w, y+h), (255,0,0),3)

        #S = w * h

       # print("x: " + str(x))
       # print("y: " + str(y))
        #print("w: " + str(w))
       # print("h: " + str(h))
       # print("S: " + str(S))

      # GoodFeaturesToTrack 

        #corners = cv2.goodFeaturesToTrack(canny,100000000,0.01,10)
        #corners = np.int0(corners)

        #for i in corners:
            #x,y = i.ravel()
            #cv2.circle(img,(x,y),3,255,-1)

       #print(x,y)

        cv2.imshow('mask', mask) 
        cv2.imshow('image_origin', image_origin)
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