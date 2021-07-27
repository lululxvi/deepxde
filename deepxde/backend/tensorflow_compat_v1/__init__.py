from __future__ import absolute_import

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow.compat.v1 as tf

# The major changes from TensorFlow 1.x to TensorFlow 2.x are:
# 1. Eager execution: enable_eager_execution(), disable_eager_execution()
# 2. Resource variables: enable_resource_variables(), disable_resource_variables()
# 3. Tensor shapes: enable_v2_tensorshape(), disable_v2_tensorshape()
# 4. Control flow: enable_control_flow_v2(), disable_control_flow_v2()
# 5. Tensors comparison: enable_tensor_equality(), disable_tensor_equality()
# 6. Some internal uses of tf.data symbols
# For more details, see
# - https://www.tensorflow.org/guide/migrate
# - the source code of disable_v2_behavior()

# We can simply disable all TensorFlow 2.x behaviors by disable_v2_behavior(), but some
# features in TensorFlow 2.x are useful such as `Tensor shapes`. Actually we use `Tensor
# shapes` in DeepXDE.
tf.disable_v2_behavior()
tf.enable_v2_tensorshape()

# In terms of functionality, we only need to disable eager mode.
# tf.disable_eager_execution()
# It hurts performance a lot (only in some cases?) if enabling tensor equality.
# tf.disable_tensor_equality()
# It hurts performance a little (only in some cases?) if enabling resource variables.
# tf.disable_resource_variables()
# It hurts performance a little (only in some cases?) if enabling control flow.
# tf.disable_control_flow_v2()

from .tensor import *

lib = tf
