import tensorflow as tf

# Automatic differentiation example
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**3 + 2*x
dy_dx = tape.gradient(y, x)   # Returns 29.0 (3x² + 2 at x=3)

print("dy/dx at x=3:", dy_dx.numpy())

# For PINNs you do this with a neural network output
# and compute ∂u/∂x, ∂²u/∂x², ∂u/∂t automatically