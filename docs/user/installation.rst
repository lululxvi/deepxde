Installation
============

DeepXDE requires `TensorFlow <https://www.tensorflow.org/>`_ to be installed. Both TensorFlow 1 and TensorFlow 2 can be used as the DeepXDE backend, but TensorFlow 1 is recommended:

- In my tests TensorFlow 2 is 2x~3x slower than TensorFlow 1;
- Currently L-BFGS optimizer is not supported in DeepXDE yet when using TensorFlow 2.

Then, you can install DeepXDE itself. If you use TensorFlow 2, you need to install DeepXDE by cloning the folder.  If you use Python 2, you need to install DeepXDE using `pip`.

- Install the stable version with ``pip``::

    $ pip install deepxde

- Install the stable version with ``conda``::

    $ conda install -c conda-forge deepxde

- For developers, you should clone the folder to your local machine and put it along with your project scripts::

    $ git clone https://github.com/lululxvi/deepxde.git

- Dependencies

    - `Matplotlib <https://matplotlib.org/>`_
    - `NumPy <http://www.numpy.org/>`_
    - `SALib <http://salib.github.io/SALib/>`_
    - `scikit-learn <https://scikit-learn.org>`_
    - `SciPy <https://www.scipy.org/>`_
    - `TensorFlow <https://www.tensorflow.org/>`_>=1.14.0
