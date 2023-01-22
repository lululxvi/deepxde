Install and Setup
=================

Installation
------------

DeepXDE requires one of the following backend-specific dependencies to be installed:

- TensorFlow 1.x: `TensorFlow <https://www.tensorflow.org>`_>=2.7.0
- TensorFlow 2.x: `TensorFlow <https://www.tensorflow.org>`_>=2.2.0, `TensorFlow Probability <https://www.tensorflow.org/probability>`_>=0.10.0
- PyTorch: `PyTorch <https://pytorch.org>`_>=1.9.0
- JAX: `JAX <https://jax.readthedocs.io>`_, `Flax <https://flax.readthedocs.io>`_, `Optax <https://optax.readthedocs.io>`_
- PaddlePaddle: `PaddlePaddle <https://www.paddlepaddle.org.cn/en>`_ (`develop version <https://www.paddlepaddle.org.cn/documentation/docs/en/install/compile/fromsource_en.html>`_)

Then, you can install DeepXDE itself.

- Install the stable version with ``pip``::

    $ pip install deepxde

- Install the stable version with ``conda``::

    $ conda install -c conda-forge deepxde

- For developers, you should clone the folder to your local machine and put it along with your project scripts::

    $ git clone https://github.com/lululxvi/deepxde.git

- Other dependencies

    - `Matplotlib <https://matplotlib.org/>`_
    - `NumPy <http://www.numpy.org/>`_
    - `scikit-learn <https://scikit-learn.org>`_
    - `scikit-optimize <https://scikit-optimize.github.io>`_
    - `SciPy <https://www.scipy.org/>`_

Docker
------

The `DeepXDE Docker image <https://hub.docker.com/r/pescapil/deepxde>`_ is configured to run DeepXDE with the GPU support. You need first to install `NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`_. Then you can run a Jupyter Notebook environment with GPU-enabled stable DeepXDE using::

    $ nvidia-docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/deepxde:latest

The `Dockerfile <https://github.com/lululxvi/deepxde/tree/master/docker/Dockerfile>`_ is based on `Horovod Docker image <https://hub.docker.com/r/horovod/horovod>`_ with TensorFlow and PyTorch. To build a DeepXDE image, you can run::

    $ git clone https://github.com/lululxvi/deepxde.git
    $ cd deepxde/docker
    $ docker build -f Dockerfile . -t deepxde

and then run your own DeepXDE image via::

$ nvidia-docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 deepxde

Working with different backends
-------------------------------

DeepXDE supports TensorFlow 1.x (``tensorflow.compat.v1`` in TensorFlow 2.x), TensorFlow 2.x, PyTorch, JAX, and PaddlePaddle backends. DeepXDE will choose the backend on the following options (high priority to low priority)

* Use the ``DDE_BACKEND`` environment variable:

    - You can use ``DDE_BACKEND=BACKEND python pde.py`` to specify the backend:

    $ DDE_BACKEND=tensorflow.compat.v1 python pde.py

    $ DDE_BACKEND=tensorflow python pde.py

    $ DDE_BACKEND=pytorch python pde.py

    $ DDE_BACKEND=jax python pde.py

    $ DDE_BACKEND=paddle python pde.py

    - Or set the global environment variable ``DDE_BACKEND`` as ``BACKEND``. In Linux, this usually can be done via ``export DDE_BACKEND=BACKEND``; in Windows, set the environment variable ``DDE_BACKEND`` in System Settings

* Modify the ``config.json`` file under "~/.deepxde":

    - The file has the content such as ``{"backend": "tensorflow.compat.v1"}``
    - You can also use ``python -m deepxde.backend.set_default_backend BACKEND`` to set the default backend
    - In Windows, you can find ``config.json`` file under "C:/Users/Username/.deepxde" directory

Currently ``BACKEND`` can be chosen from "tensorflow.compat.v1" (TensorFlow 1.x backend), "tensorflow" (TensorFlow 2.x backend), "pytorch" (PyTorch), "jax" (JAX), and "paddle" (PaddlePaddle). The default backend is TensorFlow 1.x.

Which backend should I choose?
``````````````````````````````

Although TensorFlow 1.x is the default backend, it may not be your best choice. Here is a comparison between different backends:

- Different backends support slightly different features, and switch to another backend if DeepXDE raised a backend-related error.
    - Currently, the number of features supported is: TensorFlow 1.x > TensorFlow 2.x > PyTorch > JAX.
    - Some features can be implemented easily (basically translating from one framework to another), and we welcome your contributions.
- Different backends have different computational speed, and switch to another backend if the speed is an issue in your case.
    - We find that there is no backend that is always faster than the others.
    - The speed depends on the specific problem, the dataset size, your hardware, etc. In some cases, one backend could be significantly faster than others such as 4x faster.
    - If you are not sure which one is the fastest for your problem, in general, we recommend TensorFlow 2.x, because we find that in some cases TensorFlow 2.x > PyTorch > TensorFlow 1.x (> means faster).

TensorFlow 1.x backend
``````````````````````

Export ``DDE_BACKEND`` as ``tensorflow.compat.v1`` to specify TensorFlow 1.x backend. The required TensorFlow version is 2.2.0 or later. Essentially, TensorFlow 1.x backend uses the API `tensorflow.compat.v1 <https://www.tensorflow.org/api_docs/python/tf/compat/v1>`_ in TensorFlow 2.x and disables the eager execution:

.. code:: python

   import tensorflow.compat.v1 as tf
   tf.disable_eager_execution()

In addition, DeepXDE will set ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``true`` to prevent TensorFlow take over the whole GPU memory.

TensorFlow 2.x backend
``````````````````````

Export ``DDE_BACKEND`` as ``tensorflow`` to specify TensorFlow 2.x backend. The required TensorFlow version is 2.2.0 or later. In addition, DeepXDE will set ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``true`` to prevent TensorFlow take over the whole GPU memory.

PyTorch backend
```````````````

Export ``DDE_BACKEND`` as ``pytorch`` to specify PyTorch backend. The required PyTorch version is 1.9.0 or later. In addition, if GPU is available, DeepXDE will set  the default tensor type to cuda, so that all the tensors will be created on GPU as default:

.. code:: python

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

JAX backend
```````````

Export ``DDE_BACKEND`` as ``jax`` to specify JAX backend.

PaddlePaddle backend
````````````````````

Export ``DDE_BACKEND`` as ``paddle`` to specify PaddlePaddle backend. In addition, if GPU is available, DeepXDE will set the default device to GPU, so that all the tensors will be created on GPU as default:

.. code:: python

    if paddle.device.is_compiled_with_cuda():
        paddle.device.set_device("gpu")
