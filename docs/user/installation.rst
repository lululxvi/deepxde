Install and Setup
=================

Installation
------------

DeepXDE requires one of the following backend-specific dependencies to be installed:

- TensorFlow 1.x: `TensorFlow <https://www.tensorflow.org/>`_>=2.2.0
- TensorFlow 2.x: `TensorFlow <https://www.tensorflow.org/>`_>=2.2.0 and `TensorFlow Probability <https://www.tensorflow.org/probability>`_>=0.10.0
- PyTorch: `PyTorch <https://pytorch.org/>`_

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

DeepXDE supports TensorFlow 1.x (``tensorflow.compat.v1`` in TensorFlow 2.x), TensorFlow 2.x, and PyTorch backends. DeepXDE will choose the backend on the following options (high priority to low priority)

* Use the ``DDEBACKEND`` environment variable:

    - You can use ``DDEBACKEND=BACKEND python pde.py`` to specify the backend:

    $ DDEBACKEND=tensorflow.compat.v1 python pde.py

    $ DDEBACKEND=tensorflow python pde.py

    $ DDEBACKEND=pytorch python pde.py

    - Or ``export DDEBACKEND=BACKEND`` to set the global environment variable

* Modify the ``config.json`` file under "~/.deepxde":

    - The file has the content such as ``{"backend": "tensorflow.compat.v1"}``
    - You can also use ``python -m deepxde.backend.set_default_backend BACKEND`` to set the default backend

Currently ``BACKEND`` can be chosen from "tensorflow.compat.v1" (TensorFlow 1.x backend), "tensorflow" (TensorFlow 2.x backend), and "pytorch" (PyTorch). The default backend is TensorFlow 1.x.

Which backend should I choose?
``````````````````````````````

Although TensorFlow 1.x is the default backend, it may not be your best choice. Here is a comparison between different backends:

- Different backends support slightly different features, and switch to another backend if DeepXDE raised a backend-related error.
    - Currently, the number of features supported is: TensorFlow 1.x > TensorFlow 2.x > PyTorch.
    - Some features can be implemented easily (basically translating from one framework to another), and we welcome your contributions.
- Different backends have different computational speed, and switch to another backend if the speed is an issue in your case.
    - We find that there is no backend that is always faster than the others.
    - The speed depends on the specific problem, the dataset size, your hardware, etc. In some cases, one backend could be significantly faster than others such as 4x faster.
    - If you are not sure which one is the fastest for your problem, in general, we recommend TensorFlow 2.x, because we find that in some cases TensorFlow 2.x > PyTorch > TensorFlow 1.x (> means faster).

TensorFlow 1.x backend
``````````````````````

Export ``DDEBACKEND`` as ``tensorflow.compat.v1`` to specify TensorFlow 1.x backend. The required TensorFlow version is 2.2.0 or later. Essentially, TensorFlow 1.x backend uses the API `tensorflow.compat.v1 <https://www.tensorflow.org/api_docs/python/tf/compat/v1>`_ in TensorFlow 2.x and disables the eager execution:

.. code:: python

   import tensorflow.compat.v1 as tf
   tf.disable_eager_execution()

In addition, DeepXDE will set ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``true`` to prevent TensorFlow take over the whole GPU memory.

TensorFlow 2.x backend
``````````````````````

Export ``DDEBACKEND`` as ``tensorflow`` to specify TensorFlow 2.x backend. The required TensorFlow version is 2.2.0 or later. In addition, DeepXDE will set ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``true`` to prevent TensorFlow take over the whole GPU memory.

PyTorch backend
```````````````

Export ``DDEBACKEND`` as ``pytorch`` to specify PyTorch backend. In addition, if GPU is available, DeepXDE will set  the default tensor type to cuda, so that all the tensors will be created on GPU as default:

.. code:: python

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
