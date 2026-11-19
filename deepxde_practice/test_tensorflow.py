import tensorflow as tf

print("TensorFlow version:", tf.__version__)

a = tf.constant(10)
b = tf.constant(20)

print("Result:", (a + b).numpy())