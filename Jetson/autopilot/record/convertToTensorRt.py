from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

# Get .h5 model
model_name = 'v17.1_epoch_13'
loaded_model = keras.models.load_model('/home/jetson/diypilot/notebooks/trained_models/' + model_name + '.h5')
loaded_model.summary()

# Convert to and save as .pb
pb_model_path = './trained_models/' + model_name
loaded_model.save(pb_model_path)


# Convert to and save as TensorRT .pb
output_saved_model_dir = pb_model_path + '/trt/'

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<25))
conversion_params = conversion_params._replace(precision_mode="FP16")
#conversion_params = conversion_params._replace(maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=pb_model_path,
                                    conversion_params=conversion_params)
converter.convert()

def my_input_fn():
  # Input for a single inference call, for a network that has two input tensors:
  inp1 = np.random.normal(size=(1, 170, 640, 3)).astype(np.float32)
  inp2 = np.random.normal(size=(1, 1)).astype(np.float32)
  yield [inp1, inp2]
    
converter.build(input_fn=my_input_fn)
converter.save(output_saved_model_dir)
