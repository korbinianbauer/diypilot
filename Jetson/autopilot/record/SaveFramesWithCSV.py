import pyrealsense2 as rs
import cv2
import csv
import time
import datetime
import numpy as np
import binascii
import os
import _thread

import can

import tensorflow as tf
from tensorflow import keras

from AutopilotGUI import AutopilotGUI

os.system("sudo ip link set can0 up type can bitrate 33300")
bus = can.interface.Bus('can0', bustype='socketcan')
can_dict = {}

# Own thread that continuously reads the CAN bus
def read_can(threadName):
    global can_dict
    while True:
        message = bus.recv()
        can_dict[message.arbitration_id] = message

try:
   _thread.start_new_thread( read_can, ("Thread-1", ) )
except:
   print("Error: unable to start thread")
   
   
def crop_to_roi(frame):
        roi_y = 190
        roi_h = 210
        roi_x = 0
        roi_w = 848
        crop_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        return crop_img
   
def get_steering_wheel_angle(can_dict):
    
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x10240040'):
            data = message.data
            #print("SWA:" + hex(arbitration_id) + ": " + str(binascii.hexlify(data)))
            reading = int.from_bytes(data[4:6], byteorder='big', signed=True)
            angle = round(reading*360/5750, 2)
            print("SWA: " + str(angle) + "deg")
            return angle
            
    return -999
            
def get_speed(can_dict):
    
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x106b8040'):
            data = message.data
            #print("Speed (km/h):" + hex(arbitration_id) + ": " + str(binascii.hexlify(data)))
            speed = int.from_bytes(data[4:6], byteorder='big', signed=True) * 0.031
            print("V = " + str(speed) + "km/h")
            return speed
            
    return -999
            
def access_bit(data, num):
    base = int(num // 8)
    shift = int(num%8)
    return (data[base] & (1<<shift)) >> shift
    
def get_bitfield(data):
    return [access_bit(data, i) for i in range(len(data)*8)]
    
def get_blinker(can_dict):
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x1020c040'):
            data = message.data
            blinker_links = get_bitfield(data)[5]
            blinker_rechts = get_bitfield(data)[6]
            
            print("Blink_L: " + str(blinker_links))
            print("Blink_R: " + str(blinker_rechts))
            
            return [blinker_links, blinker_rechts]
            
    return [0,0]

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)


#config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
device = profile.get_device()
roi_sensor = device.first_roi_sensor()
sensor_roi = roi_sensor.get_region_of_interest()
sensor_roi.min_x, sensor_roi.max_x = 0, 848
sensor_roi.min_y, sensor_roi.max_y = 90, 240 # oberer Teil, weil Kamera auf dem Kopf
roi_sensor.set_region_of_interest(sensor_roi)

color_sensor = device.first_color_sensor()
print(color_sensor.get_supported_options())
color_sensor.set_option(rs.option.auto_exposure_priority, 0)


#time.sleep(10)

record_timestamp = time.ctime()

# create new recording dir
record_dir = "/home/jetson/diypilot/Jetson/autopilot/records/" + record_timestamp + "/"
frame_dir = record_dir + "frames/"
os.mkdir(record_dir)
os.mkdir(frame_dir)

# Load NN
tf.config.list_physical_devices('GPU')
#model_name = 'diypilot_v9_small_FC_epoch_3'
model_name = 'diypilot_v11_full_balance_epoch_10'
loaded_model = keras.models.load_model('/home/jetson/diypilot/notebooks/trained_models/' + model_name + '.h5')
loaded_model.summary()

with open(record_dir + str(time.ctime()) + '.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    gui = AutopilotGUI()
    gui.show_window()
    gui.set_show_overlay(True)
    
    gui.set_engaged(False)

    starttime = time.time()
    timestamp = time.time()
    
    framecounter = 0
    while (time.time() - starttime) < 1800:
        last_timestamp = timestamp
        timestamp = time.time()
        mainloop_fps = 1/(timestamp - last_timestamp)
        print("Mainloop fps: " + str(round(mainloop_fps, 2)))
        frames = pipeline.wait_for_frames()
        roi_sensor.set_region_of_interest(sensor_roi)
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue


        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        
        
        # Run model
        frame_crop = crop_to_roi(color_image)
        test_data = np.expand_dims(frame_crop, axis=0)
        start = time.time()
        predicted_swa = 0
        predictions = loaded_model.predict(test_data)
        predicted_swa = predictions[0][0]*90
        print(predictions[0][0]*90)
        end = time.time()
        print("Inference took {}s".format(end-start))

        # Save frames to file and values to csv
        color_frame_filename = str(framecounter).zfill(7) + "_" + str(timestamp) + '.jpg'
        os.system('clear')
        actual_swa_deg = get_steering_wheel_angle(can_dict)
        speed = get_speed(can_dict)
        blinkers = get_blinker(can_dict)
        if (speed > 0):
            gui.set_recording(True)
            csvwriter.writerow([color_frame_filename, actual_swa_deg, speed, blinkers[0], blinkers[1]])
            cv2.imwrite(frame_dir + color_frame_filename, color_image)
        else:
            gui.set_recording(False)
        
        # GUI stuff
        t = datetime.datetime.fromtimestamp(timestamp)
        s = t.strftime('%d.%m.%Y %H:%M %S.%f')[:-3]
        gui.set_timestring(s)
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        gui.set_frame(bgr_image)
        gui.set_actual_swa(actual_swa_deg)
        gui.set_predicted_swa(predicted_swa)
        gui.set_indicator_left(blinkers[0])
        gui.set_indicator_right(blinkers[1])
        gui.set_velocity(speed)
        

    pipeline.stop()
    gui.stop_window()
