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

tf.config.list_physical_devices('GPU')

model_name = 'diypilot_v9_small_FC_epoch_3'
loaded_model = keras.models.load_model('/home/jetson/diypilot/notebooks/trained_models/' + model_name + '.h5')
loaded_model.summary()
frame = cv2.imread('/home/jetson/diypilot/notebooks/frame1.jpg')
frame_crop = crop_to_roi(frame)

test_data = np.expand_dims(frame_crop, axis=0)

yappi.start()

for i in range(10):
    start = time.time()
    predictions = loaded_model.predict(test_data)
    print(predictions[0][0]*90)
    end = time.time()
    print("Inference took {}s".format(end-start))
    
    
yappi.stop()
# retrieve thread stats by their thread id (given by yappi)
threads = yappi.get_thread_stats()
for thread in threads:
    print("Function stats for (%s) (%d)" % (thread.name, thread.id))
    yappi.get_func_stats(ctx_id=thread.id).print_all()
