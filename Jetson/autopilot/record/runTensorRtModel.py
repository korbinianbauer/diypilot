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
  image = np.asarray(image).astype(np.float32)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  return model_fn(input_tensor)

tf.config.list_physical_devices('GPU')

model_name = 'diypilot_v9_small_FC_epoch_3'
loaded_model = keras.models.load_model('/home/jetson/diypilot/Jetson/autopilot/record/trained_models/' + model_name + '/trt/')
frame = cv2.imread('/home/jetson/diypilot/notebooks/frame1.jpg')
frame_crop = crop_to_roi(frame)

yappi.start()

for i in range(1000):
    start = time.time()
    predictions = run_inference_for_single_image(loaded_model, frame_crop)
    a = tf.make_tensor_proto(predictions['dense_1'])
    predicted_swa = tf.make_ndarray(a)[0][0]*90
    #print()
    end = time.time()
    print("Inference took {}s".format(end-start))
    frame_crop = cv2.flip(frame_crop, 0)
    
yappi.stop()
# retrieve thread stats by their thread id (given by yappi)
threads = yappi.get_thread_stats()
for thread in threads:
    print("Function stats for (%s) (%d)" % (thread.name, thread.id))
    yappi.get_func_stats(ctx_id=thread.id).print_all()
