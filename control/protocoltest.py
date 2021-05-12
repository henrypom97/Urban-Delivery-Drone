import sys                                    
import traceback
#import tellopy
from tello import tellotest
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


prev_flight_data = None
run_recv_thread = True
new_image = None
flight_data = None
log_data = None

A = None

drone = tellotest.Tello()

def OpenCV():

    retry = 3
    container = None
    while container is None and 0 < retry:
      retry -= 1
      try:
      # 動画の受信開始を処理
        container = av.open(drone.get_video_stream())       # container = tello video　映像の圧縮データを展開
      except av.AVError as ave:
        print(ave)
        print('retry...')
            
    frame_skip = 300                               #動画接続前
        
    while 1:
      for frame in container.decode(video=0):         # .decode byte(映像の中身) -> 文字列
        if 0 < frame_skip: #フレームスキップ処理
          frame_skip = frame_skip - 1
          continue
            
        start_time = time.time()                   # time.time UNIX time(0:0:0)からの経過時間
          
        image_origin = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)     #RGB convert
        
        cv2.imshow("image",image_origin)

        if frame.time_base < 1.0/60:
          time_base = 1.0/60                 #機械のエラーを判別するための基準
        else:
          time_base = frame.time_base
          #フレームスキップ値を算出
          frame_skip = int((time.time() - start_time) / time_base)


def handler(event, sender, data, **args):
    global prev_flight_data
    global flight_data
    global log_data
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        if prev_flight_data != str(data):
            print(data)
            prev_flight_data = str(data)
        flight_data = data
    elif event is drone.EVENT_LOG_DATA:
        log_data = data
    else:
        print('event="%s" data=%s' % (event.getname(), str(data)))


def main():  
  try:

    drone.connect()                         #tello connection
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    drone.subscribe(drone.EVENT_LOG_DATA, handler)
    drone.wait_for_connection(60.0)

    time.sleep(5)

    #A = iter(log_data.imu)

    print(str(flight_data))
    print(log_data.imu)


    print(list(A))
        
  except Exception as ex:
    exc_type, exc_value, exc_traceback = sys.exc_info()                  # zikkoutyuuno sagyou no zyouhou teizi
    traceback.print_exception(exc_type, exc_value, exc_traceback)        # zikkoukatei deno stuck frame no kiroku wo print
    print(ex)
  finally:
    drone.land()
    drone.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
  main()