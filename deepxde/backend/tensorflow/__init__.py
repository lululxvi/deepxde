from __future__ import absolute_import

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from .tensor import *

lib = tf
