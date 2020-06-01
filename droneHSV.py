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
        img_HSV = cv2.cvtColor(image_origin, cv2.COLOR_BGR2HSV)
        img_mask = cv2.GaussianBlur(img_HSV, (19, 19), 0)
        canny = cv2.Canny(img_mask, 100, 300)
        
        #色の抽出
        
        def red_detect(canny):
        
          hsvLower1 = np.array([0, 127, 0])    # 抽出する色の下限(BGR)
          hsvUpper1 = np.array([30 ,255, 255])   # 抽出する色の上限(BGR)
          mask1 = cv2.inRange(img_mask, hsvLower1, hsvUpper1) 
        
          hsvLower2 = np.array([150, 127, 0])    # 抽出する色の下限(BGR)
          hsvUpper2 = np.array([179 ,255, 255])   # 抽出する色の上限(BGR)
          mask2 = cv2.inRange(img_mask, hsvLower2, hsvUpper2)

          return mask1 + mask2
          
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(canny)
        ax.axis("off")

        mask = red_detect(canny)

        #輪郭の抽出
        #輪郭の階層情報
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image_origin, contours, -1, (255,0,0), 5)

        #枠線の表示
        #def draw_contours(ax, image, contours):
            #ax.imshow(image)  # 画像を表示する。
            #ax.set_axis_off()
    
        #for i, cnt in enumerate(contours):
            # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
            #cnt = cnt.squeeze(axis=1)
            # 輪郭の点同士を結ぶ線を描画する。
            #ax.add_patch(Polygon(cnt, color="b", fill=None, lw=2))
            # 輪郭の点を描画する。
            #ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
            # 輪郭の番号を描画する。
            #ax.text(cnt[0][0], cnt[0][1], i, color="orange", size="10")

        #fig, ax = plt.subplots(figsize=(15, 15))
        #draw_contours(ax, image, contours)
                        
        cv2.imshow('image', image) 
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
