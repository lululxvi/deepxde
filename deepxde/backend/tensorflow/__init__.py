from __future__ import absolute_import

from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) < LooseVersion("2.2.0"):
    raise RuntimeError("DeepXDE requires tensorflow>=2.2.0.")
