import os
print("\n\n\n\nCWD:")
print(os.getcwd())
os.chdir('/home/jetson/diypilot/Jetson/autopilot/record')
print(os.getcwd())

import sys
sys.path.append('/home/jetson/Desktop/librealsense-2.39.0/build/wrappers/python')

import time

import yappi
yappi.set_clock_type("cpu")


import pyrealsense2 as rs
import cv2
import csv
import datetime
import numpy as np
import binascii
import os
import shutil 
import _thread

from threading import Thread

import can

import tensorflow as tf
from tensorflow import keras

from AutopilotGUI import AutopilotGUI

from collections import deque

stop_threads = False
can_thread = None
camera_thread = None
sample_writer_thread = None

demo_mode = False
min_frame_time = 0.066 # s, Don't re-run mainloop until this time has passed

cam_fps = 0
can_sps = 0

bus = None
can_dict = {}

# Own thread that continuously reads the CAN bus
def start_can_reader():
    if demo_mode:
        return
    can_thread = Thread(target=read_can, args=()).start()

def read_can():
    global can_dict
    global can_sps
    global bus
    last_can_time = time.time()
    while not stop_threads:
        if bus is None:
            os.system("sudo ip link set can0 up type can bitrate 33300")
            bus = can.interface.Bus('can0', bustype='socketcan')
        message = bus.recv()
        #print(message)
        can_dict[message.arbitration_id] = message
        now = time.time()
        can_sps = 1/(now - last_can_time)
        last_can_time = now
        #print("read_can()")
    print("CAN-read Thread finished")

  
start_can_reader()

#time.sleep(60)
  
camera_frame = []
camera_frame_crop = []
frame_crop_tensor = None

pipeline = None
# Own thread that continuously reads the camera frame
def read_camera():
        global camera_thread
        global pipeline
        if camera_thread is None:
            # Configure depth and color streams
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)

            # Start streaming
            profile = None
            while profile is None:
                try:
                    profile = pipeline.start(config)
                except:
                    print("Could not connect to RealSense Camera. Retrying in 1s")
                    time.sleep(1)
            device = profile.get_device()
            roi_sensor = device.first_roi_sensor()
            sensor_roi = roi_sensor.get_region_of_interest()
            sensor_roi.min_x, sensor_roi.max_x = 0, 848
            sensor_roi.min_y, sensor_roi.max_y = 90, 240 # oberer Teil, weil Kamera auf dem Kopf
            roi_sensor.set_region_of_interest(sensor_roi)

            color_sensor = device.first_color_sensor()
            print(color_sensor.get_supported_options())
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
            color_sensor.set_option(rs.option.frames_queue_size, 1)
            
            camera_thread = Thread(target=get_cam_frame, args=(), name = "Camera Thread").start()
        return
        

    
        
def get_cam_frame():
    global camera_frame
    global camera_frame_crop
    global frame_crop_tensor
    global pipeline
    global cam_fps
    
    last_frame_time = time.time()
    while not stop_threads:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        camera_frame = cv2.rotate(color_image, cv2.ROTATE_180)
        camera_frame_crop = crop_to_roi(camera_frame)
        
        image = np.asarray(camera_frame_crop).astype(np.float32)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        frame_crop_tensor = input_tensor[tf.newaxis,...]
        now = time.time()
        cam_fps = 1/(now - last_frame_time)
        #print("Shape :{}".format(camera_frame.shape))
        #print("Crop shape :{}".format(camera_frame_crop.shape))
        last_frame_time = now
        #print("get_cam_frame()")
    pipeline.stop()
    print("Camera read Thread finished")
    
   
def crop_to_roi(frame):
        roi_y = 190
        roi_h = 210
        roi_x = 0
        roi_w = 848
        crop_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        return crop_img
        
read_camera()

def stop_all_threads():
    global stop_threads
    stop_threads = True

sample_to_disk_queue = deque()

def sample_writer():
    record_timestamp = time.ctime()

    # create new recording dir
    record_dir = "/home/jetson/diypilot/Jetson/autopilot/records/" + record_timestamp + "/"
    frame_dir = record_dir + "frames/"
    os.mkdir(record_dir)
    os.mkdir(frame_dir)
    
    print("Starting sample_writer Thread")
    sample_writer_thread = Thread(target=write_sample_to_disk, args=(record_dir, frame_dir)).start()
    

def write_sample_to_disk(record_dir, frame_dir):
    global sample_to_disk_queue
    print("write_sample_to_disk()")
    with open(record_dir + str(time.ctime()) + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        while sample_to_disk_queue or not stop_threads:
            if sample_to_disk_queue:
                sample = sample_to_disk_queue.popleft() # get oldest sample
                color_frame_filename, camera_frame, actual_swa_deg, speed, blinkers = sample
                csvwriter.writerow([color_frame_filename, actual_swa_deg, speed, blinkers[0], blinkers[1]])
                cv2.imwrite(frame_dir + color_frame_filename, camera_frame)


            else:
                time.sleep(0.01)
            #print("write_sample_to_disk()")
    print("sample_writer Thread finished")
        

sample_writer()
        
def run_inference_for_single_image(model_fn):
  global frame_crop_tensor
  # Run inference
  return model_fn(frame_crop_tensor)
  
  
def get_swa_from_predictions(predictions):
    #print(predictions)
    return tf.keras.backend.get_value(predictions['dense_1'])[0][0]*90

   
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




#time.sleep(10)



# Load NN
tf.config.list_physical_devices('GPU')

#model_name = 'diypilot_v9_small_FC_epoch_3'
model_name = 'diypilot_v11_full_balance_epoch_10'

loaded_model = keras.models.load_model('/home/jetson/diypilot/Jetson/autopilot/record/trained_models/' + model_name + '/trt/')

model_fn = loaded_model.signatures['serving_default']


# Run one frame through NN for warmup
print("Running warmup")
warmup_start = time.time()
run_inference_for_single_image(model_fn)
print("Warumup done ({}s)".format(round(time.time()-warmup_start, 2)))

yappi.start()

gui = AutopilotGUI()
gui.show_window(fullscreen=True)
gui.set_show_overlay(True)
gui.set_engaged(False)

starttime = time.time()
timestamp = starttime
last_timestamp = starttime

framecounter = 0

mainloop_fps_min = 10000
mainloop_fps_max = 0

steady_frame_crop = camera_frame_crop[:]

while (time.time() - starttime) < 12000:

    
    
    
    timestamp = time.time()
    mainloop_fps = 1/(timestamp - last_timestamp)
    last_timestamp = timestamp
    
    print("\nMainloop fps: {}".format(round(mainloop_fps, 2)))
    

    # Run model
    if ((480, 848, 3)!=camera_frame.shape):
        continue
    
    if ((210, 848, 3)!=camera_frame_crop.shape):
        continue
        
    predicted_swa = 0
    inference_start = time.time()
    predictions = run_inference_for_single_image(model_fn)
    inference_done = time.time()
    predicted_swa = get_swa_from_predictions(predictions)
    prediction_extraction_done = time.time()
    
    


    # Save frames to file and values to csv
    color_frame_filename = str(framecounter).zfill(7) + "_" + str(timestamp) + '.jpg'
    #os.system('clear')
    actual_swa_deg = get_steering_wheel_angle(can_dict)
    speed = get_speed(can_dict)
    blinkers = get_blinker(can_dict)
    freespace = shutil.disk_usage("/home/jetson")[-1]/1024/1024/1024
    
    collecting_values_done = time.time()
    
    
    if (speed > 0):
        gui.set_recording(True)
        sample = color_frame_filename, camera_frame, actual_swa_deg, speed, blinkers
        sample_to_disk_queue.append(sample)
    else:
        gui.set_recording(False)
        
    writing_to_disk_done = time.time()
    
    # GUI stuff
    t = datetime.datetime.fromtimestamp(timestamp)
    s = t.strftime('%d.%m.%Y %H:%M %S.%f')[:-3]
    gui.set_timestring(s)
   
    gui.set_frame(camera_frame)
    gui.set_nn_fps(mainloop_fps)
    gui.set_cam_fps(cam_fps)
    gui.set_rec_queue_len(len(sample_to_disk_queue))
    gui.set_can_sps(can_sps)
    gui.set_freespace(freespace)
    gui.set_actual_swa(actual_swa_deg)
    gui.set_predicted_swa(predicted_swa)
    gui.set_indicator_left(blinkers[0])
    gui.set_indicator_right(blinkers[1])
    gui.set_velocity(speed)
    
    gui_updating_stuff_done = time.time()
    framecounter += 1
    
    
    print("pre_inference : {}ms".format(round(1000*(inference_start-timestamp), 2)))
    
    print("run_inference_for_single_image() : {}ms".format(round(1000*(inference_done-inference_start), 2)))
    
    print("get_swa_from_predictions() : {}ms".format(round(1000*(prediction_extraction_done-inference_done), 2)))
    
    print("collecting_values_done : {}ms".format(round(1000*(collecting_values_done-prediction_extraction_done), 2)))
    
    print("writing_to_disk : {}ms".format(round(1000*(writing_to_disk_done-collecting_values_done), 2)))
    print("gui_updating_stuff_done : {}ms".format(round(1000*(gui_updating_stuff_done-writing_to_disk_done), 2)))
    
    print("sample_to_disk_queue length: {}".format(len(sample_to_disk_queue)))
    
    mainloop_done = time.time()
    print("post_gui_stuff : {}ms".format(round(1000*(mainloop_done-gui_updating_stuff_done), 2)))
    
    print("Mainloop (incl prints) took: {}ms".format(round(1000*(mainloop_done-timestamp), 2)))
    
    while (time.time() - timestamp) < min_frame_time:
        time.sleep(0.001)
    

stop_all_threads()
gui.stop_window()
    
# retrieve thread stats by their thread id (given by yappi)
print("Mainloop ran for {} seconds".format(round(time.time() - starttime), 2))
yappi.stop()
threads = yappi.get_thread_stats()
for thread in threads:
    print("Function stats for (%s) (%d)" % (thread.name, thread.id))
    yappi.get_func_stats(ctx_id=thread.id).print_all()

