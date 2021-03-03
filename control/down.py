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
#from tellopy import tellotests

drone = tellopy.Tello()

def main():  
  
  try:
    drone.connect()                         #tello connection
    drone.wait_for_connection(60.0)

    drone.takeoff() 
    time.sleep(5)

    drone.down(15)
    time.sleep(2)
    drone.down(0)
    time.sleep(2)
    
    drone.down(30)
    time.sleep(2)
    drone.down(0)

    time.sleep(10)
    drone.up(20)
    time.sleep(2)
    drone.up(0)
    time.sleep(3)
    drone.clockwise(60)
    time.sleep(2)
    drone.clockwise(0)
    time.sleep(2)
    drone.forward(10)
    time.sleep(2)
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