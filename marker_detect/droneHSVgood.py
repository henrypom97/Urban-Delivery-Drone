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
        
        image_origin = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2HSV_FULL)
        
        # 150フレームから210フレームまで5フレームごとに切り出す
        #start_frame = 150
        #end_frame = 9000
        #interval_frames = 5
        #i = start_frame + interval_frames

        # 最初のフレームに移動して取得
        #container.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        #ret, prev_frame = container.read()

        img_mask = cv2.GaussianBlur(image_origin, (15, 15), 0)  #フィルタの中の引数
        canny = cv2.Canny(img_mask, 100, 150)

        #色の抽出
        
        def red_detect(canny):
        
          HSVLower1 = np.array([0, 50, 50])    # 抽出する色の下限(BGR)
          HSVUpper1 = np.array([20 ,255, 255])   # 抽出する色の上限(BGR)
          mask1 = cv2.inRange(image_origin, HSVLower1, HSVUpper1) 

          #HSVLower2 = np.array([202, 185, 115])
          #HSVUpper2 = np.array([255, 255, 148])
          #mask2 = cv2.inRange(image_origin,HSVLower2, HSVUpper2)

          return mask1 

        mask = red_detect(canny)

        feature_params = {"maxCorners": 200,  "qualityLevel": 0.2,  "minDistance": 12,  "blockSize": 12  }
        #  特徴点の上限数 # 閾値　（高いほど特徴点数は減る) # 特徴点間の距離 (近すぎる点は除外) 
        p0 = cv2.goodFeaturesToTrack(mask, mask=None, **feature_params)

        # 特徴点をプロットして可視化
        for p in p0:
          x,y = p.ravel()
          cv2.circle(image_origin, (x, y), 5, (0, 255, 255) , -1)

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
