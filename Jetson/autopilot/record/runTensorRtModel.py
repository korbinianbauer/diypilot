#import sys
#sys.path.append("/home/jetson/diypilot/notebooks")
import yappi
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow import keras

def crop_to_roi(frame):
        roi_y = 190
        roi_h = 210
        roi_x = 0
        roi_w = 848
        crop_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        return crop_img

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

tf.config.list_physical_devices('GPU')

model_name = 'diypilot_v9_small_FC_epoch_3'
loaded_model = keras.models.load_model('/home/rkdcpro/tf/notebooks/Autopilot/diypilot/notebooks/trained_models/' + model_name + '/trt/saved_model.pb')
frame = cv2.imread('/home/jetson/diypilot/notebooks/frame1.jpg')
frame_crop = crop_to_roi(frame)

yappi.start()

for i in range(10):
    start = time.time()
    predictions = run_inference_for_single_image(loaded_model, frame)
    print(predictions)
    end = time.time()
    print("Inference took {}s".format(end-start))
    
    
yappi.stop()
# retrieve thread stats by their thread id (given by yappi)
threads = yappi.get_thread_stats()
for thread in threads:
    print("Function stats for (%s) (%d)" % (thread.name, thread.id))
    yappi.get_func_stats(ctx_id=thread.id).print_all()
