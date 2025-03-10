import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print('GPU',tf.test.is_gpu_available())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
