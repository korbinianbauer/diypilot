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
import serial

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
#min_frame_time = 0.066 # s, Don't re-run mainloop until this time has passed
min_frame_time = 0.033 # s, Don't re-run mainloop until this time has passed

cam_fps = 0
can_sps = 0

bus = None
can_dict = {}

arduino_connection = None
arduino_out = ""
arduino_in = ""

swa_error = 0

mot_speed_act = 0
gps_lat = 0
gps_long = 0
gps_sats = 0
gps_vel = 0
gps_date = None
gps_time = None

def denormalize_swa(swa_norm):
        swa = revert_signed_log([swa_norm*4], 2)[0]
        swa_deg = swa
        return swa_deg
    
def revert_signed_log(arr, zero_gap):
    arr2 = []
    for x in arr:
        if x < 0:
            x -= np.log(zero_gap)
            x = -np.exp(abs(x))+zero_gap
            arr2.append(x)
        else:
            x += np.log(zero_gap)
            x = np.exp(abs(x))-zero_gap
            arr2.append(x)
    return np.array(arr2)


# Own thread that continuously receives from ans send to the Arduino

def start_serial_reader():
    if demo_mode:
        return
    serial_thread = Thread(target=read_arduino, args=()).start()

def read_arduino():
    global arduino_connection
    
    global mot_speed_act
    global gps_lat
    global gps_long
    global gps_sats
    global gps_vel
    global gps_date
    global gps_time
    
    while not stop_threads:
        if arduino_connection is None:
            print("Connecting to Arduino")
            arduino_connection = serial.Serial('/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0', 115200, timeout=1, write_timeout=1)
            
        input_string = arduino_connection.readline()
        #print(input_string)
        try:
            antwort = eval(input_string)
            #print("Valid string received:")
            #print(antwort)
            mot_speed_act = int(antwort["mot_vel"])
            gps_lat = float(antwort["gps_lat"])
            gps_long = float(antwort["gps_long"])
            gps_sats = int(antwort["gps_sats"])
            gps_vel = float(antwort["gps_vel"])
            gps_date = antwort["gps_date"]
            gps_time = antwort["gps_time"]
        except Exception as e:
            #print(e)
            #time.sleep(0.1)
            pass
            
start_serial_reader()


def start_serial_writer():
    if demo_mode:
        return
    serial_thread = Thread(target=send_arduino, args=()).start()

def send_arduino():
    while not stop_threads:
        while arduino_connection is None:
            time.sleep(0.1)
        
        mot_vel_setpoint = int(500*swa_error)
        
        max_vel = 10000
        
        if abs(mot_vel_setpoint) > max_vel:
            mot_vel_setpoint = 0
            
            
        cmd = "MOT," + f'{mot_vel_setpoint:06}' + ",END\n"
        arduino_connection.write(cmd.encode())
        #print(cmd)
        time.sleep(0.2)
        
start_serial_writer()
#time.sleep(1000)

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
NN_input_tensor = None

pipeline = None
# Own thread that continuously reads the camera frame
def read_camera():
        global camera_thread
        global pipeline
        if camera_thread is None:
            # Configure depth and color streams
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

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
        

def load_dummy_frame():
    global camera_frame_crop
    global NN_input_tensor
    dummy_image = cv2.imread('frame1.jpg')
    camera_frame_crop = crop_to_roi(dummy_image)
    image = np.asarray(camera_frame_crop).astype(np.float32)
    input_tensor = tf.convert_to_tensor(image)
    NN_input_tensor = input_tensor[tf.newaxis,...]
       
        
def get_cam_frame():
    global camera_frame
    global camera_frame_crop
    global NN_input_tensor
    global pipeline
    global cam_fps
    
    last_frame_time = time.time()
    while not stop_threads:
        
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        except:
            print("Failed to get frame from Camera")
        color_image = np.asanyarray(color_frame.get_data())
        camera_frame = cv2.rotate(color_image, cv2.ROTATE_180)
        camera_frame_crop = crop_to_roi(camera_frame)
        
        #cv2.imshow('camera_frame_crop', camera_frame_crop)
        #print(camera_frame_crop)
        #if cv2.waitKey(1) == ord("q"):
        #    break
        
        image = np.asarray(camera_frame_crop).astype(np.float32)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        NN_input_tensor = input_tensor[tf.newaxis,...]
        now = time.time()
        cam_fps = 1/(now - last_frame_time)
        #print("Shape :{}".format(camera_frame.shape))
        #print("Crop shape :{}".format(camera_frame_crop.shape))
        last_frame_time = now
        #print("get_cam_frame()")
    pipeline.stop()
    print("Camera read Thread finished")
    

def crop_to_roi(frame):
        roi_y = 230
        roi_h = 170
        roi_x = 104
        roi_w = 640
        crop_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        return crop_img
        



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
                color_frame_filename, camera_frame, actual_swa_deg, speed, blinkers, gps_lat, gps_long, gps_vel, gps_sats, predicted_swa = sample
                csvwriter.writerow([color_frame_filename, actual_swa_deg, speed, blinkers[0], blinkers[1], gps_lat, gps_long, gps_vel, gps_sats, predicted_swa])
                cv2.imwrite(frame_dir + color_frame_filename, camera_frame)


            else:
                time.sleep(0.01)
            #print("write_sample_to_disk()")
    print("sample_writer Thread finished")
        

sample_writer()
        
def run_inference_for_single_image(model_fn):
    global NN_input_tensor
  
    v_vehicle = get_speed(can_dict)/250
    v_vehicle = np.asarray(v_vehicle).astype(np.float32)
    v_vehicle_tensor = tf.convert_to_tensor([v_vehicle])
    v_vehicle_tensor = v_vehicle_tensor[tf.newaxis,...]
    # Run inference
    return model_fn(input_1=NN_input_tensor, input_2=v_vehicle_tensor)


def get_swa_from_predictions(predictions):
    #print(predictions)
    return tf.keras.backend.get_value(predictions['dense_3'])[0][0]

   
def get_steering_wheel_angle(can_dict):
    
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x10240040'):
            data = message.data
            #print("SWA:" + hex(arbitration_id) + ": " + str(binascii.hexlify(data)))
            reading = int.from_bytes(data[4:6], byteorder='big', signed=True)
            angle = round(reading*360/5750, 2)
            #print("SWA: " + str(angle) + "deg")
            return angle
            
    return -999
            
def get_speed(can_dict):
    
    for arbitration_id, message in can_dict.items():
        #print(hex(arbitration_id) + ": " + str(binascii.hexlify(message.data)))
        if (hex(arbitration_id) == '0x106b8040'):
            data = message.data
            #print("Speed (km/h):" + hex(arbitration_id) + ": " + str(binascii.hexlify(data)))
            speed = int.from_bytes(data[4:6], byteorder='big', signed=True) * 0.031
            #print("V = " + str(speed) + "km/h")
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
            
            #print("Blink_L: " + str(blinker_links))
            #print("Blink_R: " + str(blinker_rechts))
            
            return [blinker_links, blinker_rechts]
            
    return [0,0]




#time.sleep(10)



# Load NN
tf.config.list_physical_devices('GPU')

#model_name = 'diypilot_v9_small_FC_epoch_3'
#model_name = 'diypilot_v11_full_balance_epoch_3'
model_name = 'v17.1_epoch_13'

loaded_model = keras.models.load_model('/home/jetson/diypilot/Jetson/autopilot/record/trained_models/' + model_name + '/trt/')

model_fn = loaded_model.signatures['serving_default']


# Run dummy frame through NN for warmup
load_dummy_frame()
print("Running warmup")
warmup_start = time.time()
run_inference_for_single_image(model_fn)
print("Warmup done ({}s)".format(round(time.time()-warmup_start, 2)))

# Start real camera feed
read_camera()

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
    
    #print("\nMainloop fps: {}".format(round(mainloop_fps, 2)))
    

    # Run model
    try:
        camera_frame.shape
    except:
        continue
    
    if ((480, 848, 3)!=camera_frame.shape):
        continue
    
    if ((170, 640, 3)!=camera_frame_crop.shape):
        continue
        
    predicted_swa = 0
    inference_start = time.time()
    predictions = run_inference_for_single_image(model_fn)
    inference_done = time.time()
    normed_predicted_swa = get_swa_from_predictions(predictions)
    predicted_swa = denormalize_swa(normed_predicted_swa)
    prediction_extraction_done = time.time()
    
    


    # Save frames to file and values to csv
    color_frame_filename = str(framecounter).zfill(7) + "_" + str(timestamp) + '.jpg'
    actual_swa_deg = get_steering_wheel_angle(can_dict)
    speed = get_speed(can_dict)
    blinkers = get_blinker(can_dict)
    freespace = shutil.disk_usage("/home/jetson")[-1]/1024/1024/1024
    
    collecting_values_done = time.time()
    
    
    new_swa_error = actual_swa_deg - predicted_swa
    swa_error = 0.8*swa_error + 0.2*new_swa_error
    
    
    if (speed > 0) and (abs(speed) < 300):
        gui.set_recording(True)
        if framecounter%2 == 0:
            sample = color_frame_filename, camera_frame, actual_swa_deg, speed, blinkers, gps_lat, gps_long, gps_vel, gps_sats, predicted_swa
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
    
    gui.set_mot_speed(mot_speed_act)
    gui.set_gps_velocity(gps_vel)
    gui.set_gps_sats(gps_sats)
    gui.set_gps_lat(gps_lat)
    gui.set_gps_long(gps_long)
    
    gui_updating_stuff_done = time.time()
    framecounter += 1
    
    
    #print("pre_inference : {}ms".format(round(1000*(inference_start-timestamp), 2)))
    
    #print("run_inference_for_single_image() : {}ms".format(round(1000*(inference_done-inference_start), 2)))
    
    #print("get_swa_from_predictions() : {}ms".format(round(1000*(prediction_extraction_done-inference_done), 2)))
    
    #print("collecting_values_done : {}ms".format(round(1000*(collecting_values_done-prediction_extraction_done), 2)))
    
    #print("writing_to_disk : {}ms".format(round(1000*(writing_to_disk_done-collecting_values_done), 2)))
    #print("gui_updating_stuff_done : {}ms".format(round(1000*(gui_updating_stuff_done-writing_to_disk_done), 2)))
    
    #print("sample_to_disk_queue length: {}".format(len(sample_to_disk_queue)))
    
    mainloop_done = time.time()
    #print("post_gui_stuff : {}ms".format(round(1000*(mainloop_done-gui_updating_stuff_done), 2)))
    
    #print("Mainloop (incl prints) took: {}ms".format(round(1000*(mainloop_done-timestamp), 2)))
    
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

