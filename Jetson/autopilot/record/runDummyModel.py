#import sys
#sys.path.append("/home/jetson/diypilot/notebooks")
#import yappi
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras


tf.config.list_physical_devices('GPU')

model_name = 'dummy'
loaded_model = keras.models.load_model('/home/jetson/diypilot/notebooks/trained_models/' + model_name + '.h5')
loaded_model.summary()
test_data = np.expand_dims([[0]], axis=0)

#yappi.start()

for i in range(1000):
    start = time.time()
    predictions = loaded_model.predict(test_data)
    #print(predictions)
    end = time.time()
    print("Inference took {}s".format(end-start))
    
    
#yappi.stop()
# retrieve thread stats by their thread id (given by yappi)
#threads = yappi.get_thread_stats()
#for thread in threads:
#    print("Function stats for (%s) (%d)" % (thread.name, thread.id))
#    yappi.get_func_stats(ctx_id=thread.id).print_all()
