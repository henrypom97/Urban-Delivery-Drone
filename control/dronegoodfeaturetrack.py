import sys                                    # kikai honyaku 
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
from parse import parse
import threading 
import socket

def main():
  drone = tellopy.Tello()      #tello controller

  try:
    drone.connect()                         #tello connection
    drone.wait_for_connection(60.0)
    
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
        gray = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_BGR2GRAY)

        imagesplit1 = cv2.split(image_origin)

        #Gg = imagesplit1[1] / gray
        Gg = image_origin[:,:,1] / gray

        shape = image_origin.shape
        for x in range(0, shape[0]):
          for y in range(0, shape[1]):
            if  Gg[x,y] < 0.89: 
              Gg[x,y] = 1
            else:
              Gg[x,y] = 0
        
        G1 = np.uint8(Gg)

        feature_params = {"maxCorners": 4,  "qualityLevel": 0.5,  "minDistance": 30,  "blockSize": 5}  # tokutyoute kensyutu
        #  特徴点の上限数 # 閾値　（高いほど特徴点数は減る) # 特徴点間の距離 (近すぎる点は除外) 
        p0 = cv2.goodFeaturesToTrack(G1, mask=None, **feature_params)     
        #p0 = cv2.goodFeaturesToTrack(mask, 4, 0.5, 30)  
        p0 = np.int0(p0)
        #print(p0)

        if len(p0) >= 4:
          # 特徴点をプロットして可視化
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
                elif C -y0[n[0]] < 0 and D - y0[n[1]] > 0:
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

          #s = (d1 + d2 + d3 + d4) / 2
          #Sh = np.sqrt((s-d1)*(s-d2)*(s-d3)*(s-d4))

          #s1 = (d1 + d4 + d5) / 2
          #Sh1 = np.sqrt(s1*(s1-d1)*(s1-d4)*(s1-d5))
          
          #s2 = (d2 + d3 + d5) / 2
          #Sh2 = np.sqrt(s2*(s2-d2)*(s2-d3)*(s2-d5))

          #SH = Sh1 + Sh2

          #Sg = abs((1/2)*((a[3]-a[0])*(b[1]-b[0])-(a[1]-a[0])*(b[3]-b[0])))+abs((1/2)*((a[1]-a[2])*(b[3]-b[2])-(a[3]-a[2])*(b[1]-b[2])))
          #Sw = cv2.countNonZero(G1)
          
          S1 = d1*d2
          S2 = d1*d4
          S3 = d3*d2
          S4 = d3*d4

          c1 = (a[0]+a[2])/2
          c2 = (b[0]+b[2])/2
          c11 = int(c1)
          c21 = int(c2)
          cv2.circle(image_origin, (c11, c21), 5, (0, 255, 255) , -1)

          #line1 = cv2.line(image_origin,(c11+100,c21-100),(c11+100,c21+100),1000)
          #line2 = cv2.line(image_origin,(c11+100,c21+100),(c11-100,c21+100),1000)
          #line3 = cv2.line(image_origin,(c11-100,c21+100),(c11-100,c21-100),1000)
          #line4 = cv2.line(image_origin,(c11+100,c21-100),(c11-100,c21-100),1000)

          cy = shape[0]/2
          cy1 = shape[0]/3
          cx = shape[1]/2

          cv2.circle(image_origin, (int(cx), int(cy)), 5, (0, 255, 255) , -1)
          #with open("0.3m_S1_2020_10_17.txt", "a") as f:
          #  result = "{:.7f}\n".format(S1)
          #  f.write(result)

          #cv2.imshow('img_mask1', Gg)  
          cv2.imshow('image_origin', image_origin)
          cv2.waitKey(1)
          
          if frame.time_base < 1.0/60:
              time_base = 1.0/60
              #print("T:",time_base)
              #print("frame",frame_skip)
          else:
              time_base = frame.time_base
              #print("T:",time_base)
              # フレームスキップ値を算出
              frame_skip = (time.time() - start_time) / time_base
              #print("frame",frame_skip)

  except Exception as ex:
    exc_type, exc_value, exc_traceback = sys.exc_info()                  # zikkoutyuuno sagyou no zyouhou teizi
    traceback.print_exception(exc_type, exc_value, exc_traceback)        # zikkoukatei deno stuck frame no kiroku wo print
    print(ex)
  finally:
    drone.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
  main()
