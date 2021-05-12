import time
import sys
import tellopy
import pygame
import pygame.locals
from subprocess import Popen, PIPE
import threading
import av
import cv2.cv2 as cv2  #for avoidance of pylint error
import numpy
import time
import traceback

prev_flight_data = None
run_recv_thread = True
new_image = None
flight_data = None
log_data = None
buttons = None
speed = 100
throttle = 0.0
yaw = 0.0
pitch = 0.0
roll = 0.0

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


def update(old, new, max_delta=0.3):
    if abs(old - new) <= max_delta:
        res = new
    else:
        res = 0.0
    return res


def recv_thread(drone):
    global run_recv_thread
    global new_image
    global flight_data
    global log_data

    print('start recv_thread()')
    try:
        container = av.open(drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                if flight_data:
                    draw_text(image, 'TelloPy: joystick_and_video ' + str(flight_data), 0)
                if log_data:
                    draw_text(image, 'MVO: ' + str(log_data.mvo), -3)
                    draw_text(image, ('IMU: ' + str(log_data.imu))[0:52], -2)
                    draw_text(image, '     ' + ('IMU: ' + str(log_data.imu))[52:], -1)
                new_image = image
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)

def main():
    global buttons
    global run_recv_thread
    global new_image
    pygame.init()
    pygame.joystick.init()
    current_image = None

    try:
        js = pygame.joystick.Joystick(0)
        js.init()
        js_name = js.get_name()
        print('Joystick name: ' + js_name)
        if js_name in ('Wireless Controller', 'Sony Computer Entertainment Wireless Controller'):
            buttons = JoystickPS4
        elif js_name == 'Sony Interactive Entertainment Wireless Controller':
            buttons = JoystickPS4ALT
        elif js_name in ('PLAYSTATION(R)3 Controller', 'Sony PLAYSTATION(R)3 Controller'):
            buttons = JoystickPS3
        elif js_name in ('Logitech Gamepad F310'):
            buttons = JoystickF310
        elif js_name == 'Logitech Dual Action':
            buttons = JoystickDualAction
        elif js_name == 'Xbox One Wired Controller':
            buttons = JoystickXONE
        elif js_name == 'Microsoft X-Box One S pad':
            buttons = JoystickXONES
        elif js_name == 'Xbox Wireless Controller':
            buttons = JoystickXONES_WIRELESS
        elif js_name == 'FrSky Taranis Joystick':
            buttons = JoystickTARANIS
    except pygame.error:
        pass

    if buttons is None:
        print('no supported joystick found')
        return

    drone = tellopy.Tello()
    drone.connect()
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    drone.subscribe(drone.EVENT_LOG_DATA, handler)
    threading.Thread(target=recv_thread, args=[drone]).start()

    try:
        while 1:
            # loop with pygame.event.get() is too much tight w/o some sleep
            time.sleep(0.01)
            for e in pygame.event.get():
                handle_input_event(drone, e)
            if current_image is not new_image:
                cv2.imshow('Tello', new_image)
                current_image = new_image
                cv2.waitKey(1)
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(e)

    run_recv_thread = False
    cv2.destroyAllWindows()
    drone.quit()
    exit(1)


if __name__ == '__main__':
    main()