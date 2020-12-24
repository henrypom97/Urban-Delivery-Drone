from tellopy import tellotest
import time
import av
import numpy as np
import cv2.cv2 as cv2

def main():

    drone = tellotest.Tello()

    retry = 3
    container = None
    while container is None and 0 < retry:
      retry -= 1
      try:
      # 動画の受信開始を処理
        container = av.open(drone.get_video_stream())       # container = tello video   eizou no assyuku de-ta wo tenkai
      except av.AVError as ave:
        print(ave)
        print('retry...')
            
    frame_skip = 300                               #douga setuzokumaeno 
        
    while True:
      for frame in container.decode(video=0):         # .decode byte(eizou no nakami) -> moziretu
        if 0 < frame_skip: #フレームスキップ処理
          frame_skip = frame_skip - 1
          continue
            
        start_time = time.time()                   # time.time UNIX time(0:0:0) karano keika zikan
          
        image_origin = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)     #RGB convert
        gray = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2GRAY)
         
    A = drone.statedata()
    tof = A[0]
    print(tof)

    time.sleep(10)


if __name__ == "__main__":
    main()
