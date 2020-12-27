from AutopilotGUI import AutopilotGUI
import cv2
import time
import datetime
import math

gui = AutopilotGUI()

#dummy_frame = cv2.imread("./dummy_frame.jpeg")
dummy_frame = cv2.imread("./frame1.jpg")
dummy_frame = cv2.rotate(dummy_frame, cv2.ROTATE_180)
dummy_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB) 

gui.show_window()
time.sleep(3)

    

start_time = time.time()
i = 0
while (time.time() - start_time) < 10:
    frame = dummy_frame
    show_overlay = int(time.time())%5<4
    engaged = int(time.time())%2
    recording = int(time.time())%2
    indicator_left = int(time.time())%2
    indicator_right = int(time.time())%2
    cruise_control = int(time.time())%2
    actual_swa = math.sin(i/10)*100
    predicted_swa = math.sin((i+13)/7)*100
    show_predicted_swa = int(time.time())%3
    cc_setpoint = int(time.time()*10)%200
    velocity = int(time.time()*10)%200
    fps = int(time.time()*10)%200
    freespace = int(time.time()*10)%200
    t = datetime.datetime.now()
    s = t.strftime('%d.%m.%Y %H:%M %S.%f')[:-3]
    gui.set_timestring(s)

    gui.set_frame(frame)
    gui.set_show_overlay(show_overlay)
    gui.set_engaged(engaged)
    gui.set_recording(recording)
    gui.set_indicator_left(indicator_left)
    gui.set_indicator_right(indicator_right)
    gui.set_cruise_control(cruise_control)
    gui.set_cruise_control_setpoint(cc_setpoint)
    gui.set_velocity(velocity)
    gui.set_actual_swa(actual_swa)
    gui.set_predicted_swa(predicted_swa)
    gui.set_show_predicted_swa(show_predicted_swa)
    
    i+=1
    time.sleep(0.05)

time.sleep(5)
gui.stop_window()