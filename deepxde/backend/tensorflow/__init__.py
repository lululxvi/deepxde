from __future__ import absolute_import

import os

import tensorflow as tf

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


from .tensor import *

lib = tf
