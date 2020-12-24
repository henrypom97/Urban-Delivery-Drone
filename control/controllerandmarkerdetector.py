import sys
import traceback
from tellopy import tellopy
import av
import cv2.cv2 as cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
import datetime
from tellopy import tellotest
#import tellopy

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
      #imagesplit1 = cv2.split(image_origin)

      r, g, b = image_origin[:,:,0], image_origin[:,:,1], image_origin[:,:,2]

      #Y = 0.299*r + 0.587*g + 0.114*b
      #Cb = -0.169*r - 0.331*g + 0.500*b
      #Cr = 0.500*r - 0.419*g - 0.081*b

      #h, w, c = image_origin.shape
      #for y in range(0, h):
      #  for x in range(0, w):
      #    if  Cr[y,x] > 145 and 0 <= g < 100 and 100 <= r <= 255:
      #      Cr[y,x] = 0
      #    else:
      #      Cr[y,x] = 255
                
      G = g / gray

      shape = image_origin.shape
      for y in range(0, shape[0]):
        for x in range(0, shape[1]):
          if  G[y,x] < 0.85: 
            G[y,x] = 255
          else:
            G[y,x] = 0

      G1 = np.uint8(G)       

      feature_params = {"maxCorners": 4,  "qualityLevel": 0.5,  "minDistance": 30, "blockSize": 5}#10 }  # tokutyoute kensyutu
      #特徴点の上限数 #閾値（高いほど特徴点数は減る) # 特徴点間の距離 (近すぎる点は除外) 30
      p0 = cv2.goodFeaturesToTrack(G1, mask=None, **feature_params)             
      p0 = np.int0(p0)
      
      #特徴点をプロットして可視化
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
        c1 = (a[0]+a[2])/2
        c2 = (b[0]+b[2])/2
        c11 = int(c1)
        c21 = int(c2)
        cv2.circle(image_origin, (c11, c21), 5, (0, 255, 255) , -1)

        filename = 'telloimage' + str(frame) + '.jpg'
        cv2.imwrite(filename,image_origin)
        
        S = d1*d2
      
        cy = shape[0]/2
        cy1 = shape[0]/3
        cx = shape[1]/2

        data = [S,c1,c2,p0,cx,cy,cy1]
        return data

      if frame.time_base < 1.0/60:
        time_base = 1.0/60                 #kikai no error wo hanbetu surutame no kizyunn
      else:
        time_base = frame.time_base
        frame_skip = int((time.time() - start_time) / time_base)

def main():  
  
  try:
    drone.connect()                         #tello connection
    drone.wait_for_connection(60.0)

    drone.takeoff() 
    time.sleep(5)

    AAA = OpenCV()
    S = AAA[0]
    c1 = AAA[1]
    c2 = AAA[2]
    p0 = AAA[3]
    cx = AAA[4]
    cy = AAA[5]
    cy1 = AAA[6]

    while True:
      if S <= 1000: 
        if cx - 150 > c1:
          dir = 1
        elif cx + 150 < c1:
          dir = 2
        else:
          if cy1 - 150 > c2 :
            dir = 3
          elif cy1 + 150 < c2:             
            dir = 4
          else:
            dir = 5        
      elif 1000 < S <= 4500 :
        if cx - 150 > c1:
          dir = 1
        elif cx + 150 < c1:
          dir = 2            
        else:
          if cy1 - 150 > c2 :
            dir = 3
          elif cy1 + 150 < c2:             
            dir = 4
          else:
            dir = 6        
      elif 4500 < S <= 20000:
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
      elif 20000 < S <= 80000:
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
      elif 80000 < S:
        if cx - 80 > c1:
          dir = 1
        elif cx + 80 < c1:
          dir = 2
        else:
          if cy - 80 > c2 :
            dir = 3
          elif cy + 80 < c2:             
            dir = 4
          else:
            dir = 7            

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
      elif dir == 7:
        drone.down(15)
        time.sleep(2)
        drone.down(0)
        time.sleep(10)
        drone.up(20)
        time.sleep(2)
        drone.up(0)
        time.sleep(3)
        drone.clockwise(2)
        time.sleep(2)
        drone.clockwise(0)
        drone.forward(10)
        drone.land()
        drone.quit()         
  
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