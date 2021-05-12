import sys                                    
import traceback
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
        
    while True:
      for frame in container.decode(video=0):         # .decode byte(映像の中身) -> 文字列
        if 0 < frame_skip: #フレームスキップ処理
          frame_skip = frame_skip - 1
          continue
            
        start_time = time.time()      
            
   
        image_origin = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)     #RGB convert

        h, w, c = image_origin.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        r = image[:,:,2]
        g = image[:,:,1]
        b = image[:,:,0]

        R = np.array(r).flatten()
        G = np.array(g).flatten()
        B = np.array(b).flatten()

        #R = [x for x in R if x > 15]
        #G = [x for x in G if x > 15]
        #B = [x for x in B if x > 15]

        V1 = np.std(R)
        V2 = np.std(G)
        V3 = np.std(B)

        mode = sstats.mode(R)[0]
        mode1 = sstats.mode(G)[0]
        mode2 = sstats.mode(B)[0]

        threshold_img = image.copy()


        threshold_img[r < mode1 - 3.7*V1] = 0

        threshold_img[r >= mode1 - 3.7*V1] = 255

        threshold_img[r > mode1 + 3.7*V1] = 0

        threshold_img[g > mode2 - 3.7*V2] = 0

        #feature_params = {"maxCorners": 4,  "qualityLevel": 0.3,  "minDistance": 30,  "blockSize": 12}

        #feature_params = {"maxCorners": 8,  "qualityLevel": 0.3,  "minDistance": 10,  "blockSize": 12}

        feature_params = {"maxCorners": 12,  "qualityLevel": 0.3,  "minDistance": 5,  "blockSize": 9}

        #feature_params = {"maxCorners": 4,  "qualityLevel": 0.3,  "minDistance": 5,  "blockSize": 9}
        #特徴点の上限数 # 閾値　（高いほど特徴点数は減る) # 特徴点間の距離 (近すぎる点は除外) 


        image2 = cv2.blur(image, (3, 3))
        A = np.uint8(image[:,:,2])


        image2 = cv2.blur(threshold_img, (3, 3))
        A = np.uint8(image2[:,:,2])

        p0 = cv2.goodFeaturesToTrack(A, mask=None, **feature_params)


        for p in p0:               #p0 x,y zahyou 3image_origin
          x,y = p.ravel()                                             #p0 no youso wo bunkai
          cv2.circle(image, (x, y), 5, (0, 255, 255) , -1)


        """
        cv2.imshow("image", image)
        cv2.imshow("image1", image1)
        cv2.imshow("image2", image2)
        cv2.imshow("image_thresh", threshold_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        x0 = p0[:,:,0].ravel()                                             #x座標
        y0 = p0[:,:,1].ravel()                                             #y座標


        MX = max(x0)
        mx = min(x0)

        MY = max(y0)
        my = min(y0)

        avex = (MX + mx) / 2
        avey = (MY + my) / 2


        """
        print(x0)
        print(y0)

        print(MX)
        print(mx)

        print(MY)
        print(my)

        print(avex)
        print(avey)

        print(type(avex))
        print(type(x0[0]))
        """


        MX = int(MX)
        mx = int(mx)

        MY = int(MY)
        my = int(my)

        avex = int(avex)
        avey = int(avey)

        a = [0]*12
        b = [0]*12


        x1 = []
        y1 = []
        for i in range(len(x0)):
          if y0[i] < avey:
              x1.append(x0[i])
              y1.append(y0[i])

        l11 = np.sqrt((x1[0])**2+(y1[0])**2)
        l21 = np.sqrt((x1[1])**2+(y1[1])**2)
        l31 = np.sqrt((x1[2])**2+(y1[2])**2)
        l41 = np.sqrt((x1[3])**2+(y1[3])**2)

        l1 = [l11, l21, l31, l41]

        print(l1)

        c = [0, 1, 2, 3]

        for i in range(len(l1)):
          if l1[i] == min(l1):
              a[0] = x1[i]
              b[0] = y1[i]
              s = i
        c.remove(s)
        j=0
        for j in c:
          n = c.copy()
          A = (b[0]-y1[j])/(a[0]-x1[j])
          B = b[0] - A*a[0]
          n.remove(j)
          C = A*x1[n[0]] + B

          D=A*x1[n[1]]+B
          if C -y1[n[0]] > 0 and D - y1[n[1]] < 0:
              a[1] =  x1[n[0]]
              b[1] =  y1[n[0]]
              a[3] =  x1[n[1]]
              b[3] =  y1[n[1]]
              a[2] =  x1[j]
              b[2] =  y1[j]
              break
          elif C -y1[n[0]] < 0 and D - y1[n[1]] > 0:
              a[3] =  x1[n[0]]
              b[3] =  y1[n[0]]
              a[1] =  x1[n[1]]
              b[1] =  y1[n[1]]
              a[2] =  x1[j]
              b[2] =  y1[j]
              break


        d1 = np.sqrt((a[0]-a[1])**2+(b[0]-b[1])**2)
        d2 = np.sqrt((a[1]-a[2])**2+(b[1]-b[2])**2)
        d3 = np.sqrt((a[2]-a[3])**2+(b[2]-b[3])**2)
        d4 = np.sqrt((a[3]-a[0])**2+(b[3]-b[0])**2)

        line1 = cv2.line(image,(a[0],b[0]),(a[1],b[1]),100)
        line2 = cv2.line(image,(a[1],b[1]),(a[2],b[2]),100)
        line3 = cv2.line(image,(a[2],b[2]),(a[3],b[3]),100)
        line4 = cv2.line(image,(a[3],b[3]),(a[0],b[0]),100)


        x2 = []
        y2 = []
        for i in range(len(x0)):
          if y0[i] > avey and x0[i] < avex:
              x2.append(x0[i])
              y2.append(y0[i])

        l12 = np.sqrt((x2[0])**2+(y2[0])**2)
        l22 = np.sqrt((x2[1])**2+(y2[1])**2)
        l32 = np.sqrt((x2[2])**2+(y2[2])**2)
        l42 = np.sqrt((x2[3])**2+(y2[3])**2)

        l2 = [l12, l22, l32, l42]

        print(l2)

        d = [0, 1, 2, 3]

        for i in range(len(l2)):
          if l2[i] == min(l2):
              a[4] = x2[i]
              b[4] = y2[i]
              s = i
        d.remove(s)
        k=0
        for k in d:
          n = d.copy()
          A = (b[4]-y2[k])/(a[4]-x2[k])
          B = b[4] - A*a[4]
          n.remove(k)
          C = A*x2[n[0]] + B

          D=A*x2[n[1]]+B
          if C -y2[n[0]] > 0 and D - y2[n[1]] < 0:
              a[5] =  x2[n[0]]
              b[5] =  y2[n[0]]
              a[7] =  x2[n[1]]
              b[7] =  y2[n[1]]
              a[6] =  x2[k]
              b[6] =  y2[k]
              break
          elif C -y2[n[0]] < 0 and D - y2[n[1]] > 0:
              a[5] =  x2[n[0]]
              b[5] =  y2[n[0]]
              a[6] =  x1[n[1]]
              b[6] =  y1[n[1]]
              a[7] =  x1[k]
              b[7] =  y1[k]
              break


        d4 = np.sqrt((a[4]-a[5])**2+(b[4]-b[5])**2)
        d5 = np.sqrt((a[5]-a[6])**2+(b[5]-b[6])**2)
        d6 = np.sqrt((a[6]-a[7])**2+(b[6]-b[7])**2)
        d7 = np.sqrt((a[7]-a[4])**2+(b[7]-b[4])**2)

        line4 = cv2.line(image,(a[4],b[4]),(a[5],b[5]),100)
        line5 = cv2.line(image,(a[5],b[5]),(a[6],b[6]),100)
        line6 = cv2.line(image,(a[6],b[6]),(a[7],b[7]),100)
        line7 = cv2.line(image,(a[7],b[7]),(a[4],b[4]),100)



        x3 = []
        y3 = []
        for i in range(len(x0)):
          if y0[i] > avey and x0[i] > avex:
              x3.append(x0[i])
              y3.append(y0[i])

        l13 = np.sqrt((x3[0])**2+(y3[0])**2)
        l23 = np.sqrt((x3[1])**2+(y3[1])**2)
        l33 = np.sqrt((x3[2])**2+(y3[2])**2)
        l43 = np.sqrt((x3[3])**2+(y3[3])**2)

        l3 = [l13, l23, l33, l43]

        print(l3)

        e = [0, 1, 2, 3]

        for i in range(len(l3)):
          if l3[i] == min(l3):
              a[8] = x3[i]
              b[8] = y3[i]
              s = i
        e.remove(s)
        z=0
        for z in e:
          n = e.copy()
          A = (b[8]-y3[z])/(a[8]-x3[z])
          B = b[8] - A*a[8]
          n.remove(z)
          C = A*x3[n[0]] + B

          D= A*x3[n[1]] + B
          
          #if C - y3[n[0]] > 0 and D - y3[n[1]] < 0:
          if C -y3[n[0]] < 0 and D - y3[n[1]] > 0:
              a[9] =  x3[n[0]]
              b[9] =  y3[n[0]]
              a[11] =  x3[n[1]]
              b[11] =  y3[n[1]]
              a[10] =  x3[z]
              b[10] =  y3[z]
              break
          
          #elif C - y3[n[0]] < 0 and D - y3[n[1]] > 0:
          elif C - y3[n[0]] > 0 and D - y3[n[1]] < 0:
              a[9] =  x3[n[0]]
              b[9] =  y3[n[0]]
              a[10] =  x3[n[1]]
              b[10] =  y3[n[1]]
              a[11] =  x3[z]
              b[11] =  y3[z]
              break

        d8 = np.sqrt((a[8]-a[9])**2+(b[8]-b[9])**2)
        d9 = np.sqrt((a[9]-a[10])**2+(b[9]-b[10])**2)
        d10 = np.sqrt((a[10]-a[11])**2+(b[10]-b[11])**2)
        d11 = np.sqrt((a[11]-a[8])**2+(b[11]-b[8])**2)

        line8 = cv2.line(image,(a[8],b[8]),(a[9],b[9]),100)
        line9 = cv2.line(image,(a[9],b[9]),(a[10],b[10]),100)
        line10 = cv2.line(image,(a[10],b[10]),(a[11],b[11]),100)
        line11 = cv2.line(image,(a[11],b[11]),(a[8],b[8]),100)

        """
        print(avex)
        print(avey)

        print(x0)
        print(y0)
        print('x1[0]:%d' % x1[0])
        print('y1[0]:%d' % y1[0])
        print('x1[1]:%d' % x1[1])
        print('y1[1]:%d' % y1[1])
        print('x1[2]:%d' % x1[2])
        print('y1[2]:%d' % y1[2])
        print('x1[3]:%d' % x1[3])
        print('y1[3]:%d' % y1[3])

        print('a[0]:%d' % a[0])
        print('b[0]:%d' % b[0])
        print('a[1]:%d' % a[1])
        print('b[1]:%d' % b[1])
        print('a[2]:%d' % a[2])
        print('b[2]:%d' % b[2])
        print('a[3]:%d' % a[3])
        print('b[3]:%d' % b[3])

        print('a[4]:%d' % a[4])
        print('b[4]:%d' % b[4])
        print('a[5]:%d' % a[5])
        print('b[5]:%d' % b[5])
        print('a[6]:%d' % a[6])
        print('b[6]:%d' % b[6])
        print('a[7]:%d' % a[7])
        print('b[7]:%d' % b[7])


        print('a[8]:%d' % a[8])
        print('b[8]:%d' % b[8])
        print('a[9]:%d' % a[9])
        print('b[9]:%d' % b[9])
        print('a[10]:%d' % a[10])
        print('b[10]:%d' % b[10])
        print('a[11]:%d' % a[11])
        print('b[11]:%d' % b[11])

        cv2.imshow("image", image)
        cv2.imshow("thresh", threshold_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        c1 = (a[0]+a[2])/2
        c2 = (b[0]+b[2])/2
        c3 = (a[4]+a[6])/2
        c4 = (b[4]+b[6])/2
        c5 = (a[8]+a[10])/2
        c6 = (b[8]+b[10])/2

        print(x0)
        print(y0)

        c11 = int(c1)
        c12 = int(c2)
        c13 = int(c3)
        c14 = int(c4)
        c15 = int(c5)
        c16 = int(c6)

        #line1 = cv2.line(image,(c11,c12),(c13,c14),100)
        line = cv2.line(image,(c13,c14),(c15,c16),100)
        #line3 = cv2.line(image,(c15,c16),(c11,c12),100)

        D = np.sqrt((c3 - c5)**2+(c4 - c6)**2)

        Tx = (c3 + c5)/2
        Ty = (c4 + c6)/2

        Da = (c4 - c6)/(c3 - c5)
        DDa = - 1 / Da

        Dx = (c6 - Da*c4 + DDa*c1 - c2)/(DDa - Da)
        Dy = Da*Dx + c6 - Da * c4

        #lineT = cv2.line(image,(int(Tx),int(Ty)),(c11,c12),100)

        lineDT = cv2.line(image,(int(Dx),int(Dy)),(c11,c12),100)

        T = np.sqrt((Tx - c1)**2+(Ty - c2)**2)
        DD = np.sqrt((Dx - c1)**2+(Dy - c2)**2)

        X = (T*0.6)/D

        XD = (DD * 0.6)/D

        print(DD)
        print(XD)

        print(T)
        print(D)
        print(X)


        """
        l1 = np.sqrt((c11-c13)**2 + (c12-c14)**2)
        l2 = np.sqrt((c13-c14)**2 + (c15-c16)**2)
        l3 = np.sqrt((c15-c16)**2 + (c11-c13)**2)

        s = (l1 + l2 + l3) / 2
        Sh = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))

        sin1 = Sh / (l1*l3) 
        theta1 = math.degrees(math.asin(sin1))

        sin2 = Sh / (l2*l3)
        theta2 = math.degrees(math.asin(sin2))

        sin3 = Sh / (l2*l1)
        theta3 = math.degrees(math.asin(sin3))

        print(Sh)
        print(l1)
        print(l2)
        print(l3)
        print(theta1)
        print(theta2)
        print(theta3)
        """

        S = abs((1/2)*((a[3]-a[0])*(b[1]-b[0])-(a[1]-a[0])*(b[3]-b[0])))+abs((1/2)*((a[1]-a[2])*(b[3]-b[2])-(a[3]-a[2])*(b[1]-b[2])))

        filename = 'telloimage' + str(frame) + '.jpg'
        cv2.imwrite(filename,image_origin)
        
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

    drone.takeoff() 
    time.sleep(5)

    while True:   #not need

      AAA = OpenCV()
      S = AAA[0]
      c1 = AAA[1]
      c2 = AAA[2]
      p0 = AAA[3]
      cx = AAA[4]
      cy = AAA[5]

      a = str(flight_data)

      print(a)

      with open("height protocol 2021-4-28 -1 .txt", "a") as f:
          result = "{:s}\n".format(a)
          f.write(result)


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
        drone.up(20)
        time.sleep(10)
        drone.up(0)

      
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