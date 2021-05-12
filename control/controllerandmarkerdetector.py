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
from tellopy import tellopy

drone = tellopy.Tello()

def OpenCV():

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
        #th, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        
        r, g, b = image_origin[:,:,0], image_origin[:,:,1], image_origin[:,:,2]

        h, w, c = image_origin.shape

        G = g / gray

        #shape = image_origin.shape
        for y in range(0, h):
          for x in range(0, w):
            if  G[y,x] < 0.85: 
              G[y,x] = 255
            else:
              G[y,x] = 0

        G1 = np.uint8(G)       

        feature_params = {"maxCorners": 4,  "qualityLevel": 0.5,  "minDistance": 30, "blockSize": 5}#10 }  # tokutyoute kensyutu
        #  特徴点の上限数 # 閾値　（高いほど特徴点数は減る) # 特徴点間の距離 (近すぎる点は除外) 30
        p0 = cv2.goodFeaturesToTrack(G1, mask=None, **feature_params)             
        p0 = np.int0(p0)
        
        # 特徴点をプロットして可視化
        if len(p0) == 4:
          for p in p0:               #p0 x,y zahyou 3image_origin
            x,y = p.ravel()                                             #p0 no youso wo bunkai
            cv2.circle(image_origin, (x, y), 5, (0, 255, 255) , -1)
                    
            x0 = p0[:,:,0].ravel()                                             #x zahyou 
            y0 = p0[:,:,1].ravel()                                             #y zahyou
            l1 = np.sqrt((x0[0])**2+(y0[0])**2)
            l2 = np.sqrt((x0[1])**2+(y0[1])**2)
            l3 = np.sqrt((x0[2])**2+(y0[2])**2)
            l4 = np.sqrt((x0[3])**2+(y0[3])**2)
                    
            l = [l1, l2, l3, l4]

            a = [0]*4
            b = [0]*4
            nn = [0, 1, 2, 3]
            for i in range(len(l)):
              if l[i] == min(l):
                a[0] = x0[i]
                b[0] = y0[i]
                s = i
            nn.remove(s)
            j=0
            for j in nn:
              n=nn.copy()
              A = (b[0]-y0[j])/(a[0]-x0[j])
              B = b[0] - A*a[0]
              n.remove(j)
              C = A*x0[n[0]] + B
              D = A*x0[n[1]] + B
              if C - y0[n[0]] > 0 and D - y0[n[1]] < 0:
                a[1] =  x0[n[0]]
                b[1] =  y0[n[0]]
                a[3] =  x0[n[1]]
                b[3] =  y0[n[1]]
                a[2] =  x0[j]
                b[2] =  y0[j]
                break
              elif C - y0[n[0]] < 0 and D - y0[n[1]] > 0:
                a[3] =  x0[n[0]]
                b[3] =  y0[n[0]]
                a[1] =  x0[n[1]]
                b[1] =  y0[n[1]]
                a[2] =  x0[j]
                b[2] =  y0[j]
                break

          d1 = np.sqrt((a[0]-a[1])**2+(b[0]-b[1])**2)
          d2 = np.sqrt((a[1]-a[2])**2+(b[1]-b[2])**2)
          d3 = np.sqrt((a[2]-a[3])**2+(b[2]-b[3])**2)
          d4 = np.sqrt((a[3]-a[0])**2+(b[3]-b[0])**2)
          line1 = cv2.line(image_origin,(a[0],b[0]),(a[1],b[1]),1000)
          line2 = cv2.line(image_origin,(a[1],b[1]),(a[2],b[2]),1000)
          line3 = cv2.line(image_origin,(a[2],b[2]),(a[3],b[3]),1000)
          line4 = cv2.line(image_origin,(a[3],b[3]),(a[0],b[0]),1000)
        
          #tyuuten
          c1 = (a[0]+a[2]) / 2
          c2 = (b[0]+b[2]) / 2
          c11 = int(c1)
          c21 = int(c2)
          cv2.circle(image_origin, (c11, c21), 5, (0, 255, 255) , -1)

          filename = 'telloimage' + str(frame) + '.jpg'
          cv2.imwrite(filename,image_origin)
          
          #S = cv2.countNonZero(G1)

          #s1 = (d1 + d4 + d5) / 2
          #Sh1 = np.sqrt(s1*(s1-d1)*(s1-d4)*(s1-d5))
          
          #s2 = (d2 + d3 + d5) / 2
          #Sh2 = np.sqrt(s2*(s2-d2)*(s2-d3)*(s2-d5))

          #S = Sh1 + Sh2

          S = abs((1/2)*((a[3]-a[0])*(b[1]-b[0])-(a[1]-a[0])*(b[3]-b[0])))+abs((1/2)*((a[1]-a[2])*(b[3]-b[2])-(a[3]-a[2])*(b[1]-b[2])))

          with open("S.txt", "a") as f:
            result = "{:.7f}\n".format(S)
            f.write(result)

          cy = h / 2
          cx = w / 2

          data = [S,c1,c2,p0,cx,cy]
          return data

        if frame.time_base < 1.0/60:
          time_base = 1.0/60                 #kikai no error wo hanbetu surutame no kizyunn
        else:
          time_base = frame.time_base
          #フレームスキップ値を算出
          frame_skip = int((time.time() - start_time) / time_base)

file = None
write_header = True
def handler(event, sender, data, **args):
    global file
    global write_header
    drone = sender
    if event is drone.EVENT_LOG_DATA:
        if file == None:
            path = '/Volumes/TSPANN/projects/nifi-drone/logs/tello-%s.csv' % (
                datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            file = open(path, 'w')
        if write_header:
            file.write('%s\n' % data.format_cvs_header())
            write_header = False
        file.write('%s\n' % data.format_cvs())
    if event is drone.EVENT_FLIGHT_DATA or event is drone.EVENT_LOG_DATA:
        print('record_log: %s: %s' % (event.name, str(data)))

def main():  
  drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
  drone.subscribe(drone.EVENT_LOG_DATA, handler)
  drone.record_log_data()
  
  try:
    drone.connect()                         #tello connection
    drone.wait_for_connection(60.0)

    drone.takeoff() 
    time.sleep(5)

    while True:   

      AAA = OpenCV()
      S = AAA[0]
      c1 = AAA[1]
      c2 = AAA[2]
      p0 = AAA[3]
      cx = AAA[4]
      cy = AAA[5]

      if S <= 1500: 
        if cx - 200 > c1:
          dir = 1
        elif cx + 200 < c1:
          dir = 2
        else:
          if cy - 200 > c2:
            dir = 3
          elif cy + 200 < c2:             
            dir = 4
          else:
            dir = 5
      elif 1500 < S <= 5000:
        if cx - 200 > c1:
          dir = 1
        elif cx + 200 < c1:
          dir = 2            
        else:
          if cy - 100 > c2 :
            dir = 3
          elif cy + 100 < c2:             
            dir = 4
          else:
            dir = 5
      elif 5000 < S <= 10000:
        if cx - 100 > c1:
          dir = 1
        elif cx + 100 < c1:
          dir = 2
        else:
          if cy - 100 > c2 :
            dir = 3
          elif cy + 100 < c2:             
            dir = 4
          else:
            dir = 5
      elif 10000 < S <= 30000:
        if cx - 100 > c1:
          dir = 1
        elif cx + 100 < c1:
          dir = 2
        else:
          if cy - 100 > c2 :
            dir = 3
          elif cy + 100 < c2:             
            dir = 4
          else:
            dir = 7
      #elif 30000 < S < 50000:
      #  if cx - 80 > c1:
      #    dir = 1
      #  elif cx + 80 < c1:
      #    dir = 2
      #  else:
      #    if cy - 80 > c2 :
      #      dir = 3
      #    elif cy + 80 < c2:             
      #      dir = 4
      #    else:
      #      dir = 7            

      if dir == 1:
        drone.left(5)
        time.sleep(3)
        drone.left(0)
        time.sleep(2)
      elif dir == 2:
        drone.right(5)
        time.sleep(3)
        drone.right(0)
        time.sleep(2)
      elif dir == 3:
        drone.up(10)
        time.sleep(3)
        drone.up(0)
        time.sleep(3)
      elif dir == 4:
        drone.down(10)
        time.sleep(3)
        drone.down(0)
        time.sleep(3)
      elif dir == 5:
        drone.forward(10)
        time.sleep(3)
        drone.forward(0)
        time.sleep(3)
      elif dir == 6:
        drone.forward(5)
        time.sleep(3)
        drone.forward(0)
        time.sleep(3)
      elif dir == 7:
        drone.down(30)
        time.sleep(10)
        drone.down(0)
        time.sleep(20)
        #drone.up(20)
        #time.sleep(10)
        #drone.up(0)
        #time.sleep(5)
        #drone.land()

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