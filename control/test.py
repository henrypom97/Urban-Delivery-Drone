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

tello = DJItellopy.Tello()

def OpenCV():

    tello.connect()

    tello.streamon()
    frame_read = tello.get_frame_read()
            
    frame_skip = 300                               #動画接続前
        
    while True:
        if 0 < frame_skip: #フレームスキップ処理
          frame_skip = frame_skip - 1
          continue
            
        start_time = time.time()                   # time.time UNIX time(0:0:0)からの経過時間

        img = frame_read.frame
          
        image_origin = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)     #RGB convert
        
        r = image_origin[:,:,2]
        g = image_origin[:,:,1]
        b = image_origin[:,:,0]

        R = np.array(r).flatten()
        G = np.array(g).flatten()
        B = np.array(b).flatten()

        R = [x for x in R if x > 15]
        G = [x for x in G if x > 15]
        B = [x for x in B if x > 15]

        V1 = np.std(R)
        V2 = np.std(G)
        V3 = np.std(B)

        mode = sstats.mode(R)[0]
        mode1 = sstats.mode(G)[0]
        mode2 = sstats.mode(B)[0]

        h, w, c = image_origin.shape
        for y in range(0, h, 2):
            for x in range(0, w ,2):      
                if  mode - 4.7*V1　< image_origin[y,x,2] < mode + 4.8*V1:
                    if  mode1 - 4.8*V2 < image_origin[y,x,1] < mode1 + 4.8*V2 and mode2 - 4.8*V3 < image_origin[y,x,0] < mode2 + 4.8*V3:
                        image_origin[y,x] = 0
                        image_origin[y,x+1] = 0
                        image_origin[y+1,x+1] = 0
                    else:
                        image_origin[y,x] = 255
                else:
                    image_origin[y,x+1] = 0
                    image_origin[y,x] = 0

        image_origin = cv2.blur(image_origin, (3, 3))

        A = np.uint8(image_origin[:,:,2])

        feature_params = {"maxCorners": 4,  "qualityLevel": 0.5,  "minDistance": 30, "blockSize": 5}#10 }  #特徴点検出
        #  特徴点の上限数 # 閾値　（高いほど特徴点数は減る) # 特徴点間の距離 (近すぎる点は除外) 30
        p0 = cv2.goodFeaturesToTrack(A, mask=None, **feature_params)             
        p0 = np.int0(p0)
        
        # 特徴点をプロットして可視化
        if len(p0) == 4:
          for p in p0:               #p0 x,y座標
            x,y = p.ravel()                                             #p0の要素を分解 no youso wo bunkai
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
        
          #中点
          c1 = (a[0]+a[2]) / 2
          c2 = (b[0]+b[2]) / 2
          c11 = int(c1)
          c21 = int(c2)
          cv2.circle(image_origin, (c11, c21), 5, (0, 255, 255) , -1)

          filename = 'telloimage' + str(frame) + '.jpg'
          cv2.imwrite(filename,image_origin)
          
          #S = cv2.countNonZero(G1)
          S = abs((1/2)*((a[3]-a[0])*(b[1]-b[0])-(a[1]-a[0])*(b[3]-b[0])))+abs((1/2)*((a[1]-a[2])*(b[3]-b[2])-(a[3]-a[2])*(b[1]-b[2])))

          with open("S1 2021.3.10 8:19.txt", "a") as f:
            result = "{:.7f}\n".format(S)
            f.write(result)

          with open("d1 2021.3.10 8:19..txt", "a") as f:
            result = "{:.7f}\n".format(S)
            f.write(result)

          with open("d2 2021.3.10 8:19..txt", "a") as f:
            result = "{:.7f}\n".format(d2)
            f.write(result)


          print(S)
          print(d1)
          print(d2)


          cy = h / 2
          cx = w / 2

          data = [S,c1,c2,p0,cx,cy]
          return data

        if frame.time_base < 1.0/60:
          time_base = 1.0/60                 #機械のエラーを判別するための基準
        else:
          time_base = frame.time_base
          #フレームスキップ値を算出
          frame_skip = int((time.time() - start_time) / time_base)

def main():  
  try:

    drone.connect()                         #tello connection

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
      elif 15000 < S <= 30000:
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
        tello.move_left(5)
        time.sleep(3)
        tello.move_left(0)
        time.sleep(2)
      elif dir == 2:
        tello.move_right(5)
        time.sleep(3)
        drone.move_right(0)
        time.sleep(2)
      elif dir == 3:
        drone.move_up(10)
        time.sleep(3)
        drone.move_up(0)
        time.sleep(3)
      elif dir == 4:
        drone.move_down(10)
        time.sleep(3)
        drone.move_down(0)
        time.sleep(3)
      elif dir == 5:
        drone.move_forward(10)
        time.sleep(3)
        drone.move_forward(0)
        time.sleep(3)
      elif dir == 6:
        drone.move_forward(5)
        time.sleep(3)
        drone.move_forward(0)
        time.sleep(3)
      elif dir == 7:
        drone.move_down(30)
        time.sleep(10)
        drone.move_down(0)
        time.sleep(40)
        drone.move_up(20)
        time.sleep(10)
        drone.move_up(0)
        

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