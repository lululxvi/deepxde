Antiderivative operator from an unaligned dataset
=================================================

Problem setup
-------------

We will learn the antiderivative operator

.. math:: G: v \mapsto u

defined by an ODE

.. math:: \frac{du(x)}{dx} = v(x), \qquad x \in [0, 1],

with IC :math:`u(0) = 0`.

We learn :math:`G` from a dataset. Each data point in the dataset is one triple of :math:`(v, x, u(x))`, generated as follows:

1. A random function :math:`v` is sampled from a Gaussian random field (GRF) with the resolution :math:`m = 100`.
2. Solve :math:`u` for :math:`v` numerically. We assume that for each :math:`u`, we have the value of :math:`u(x)` in only one random location :math:`x`. Because for different :math:`u`, we have the value of :math:`u(x)` in different random locations, we call this dataset as "unaligned data".

The datasets can be found at `here <https://drive.google.com/drive/folders/1PbJ2FQp8IeKhAyzOVcKeOEex7mHKyyCX?usp=sharing>`_.

- The training dataset has size 10000.
- The testing dataset has size 100000.

Implementation
--------------

To use DeepONet, we need to organize the dataset in the following format:

- Input of the branch net: the functions :math:`v`. It is a matrix of shape (dataset size, :math:`m`), e.g., (10000, 100) for the training dataset.
- Input of the trunk net: the locations :math:`x` of :math:`u(x)` values. It is a matrix of shape (dataset size, dimension), i.e., (10000, 1) for the training dataset.
- Output: The values of :math:`u(x)` in different locations for different :math:`v`. It is a matrix of shape (dataset size, 1), e.g., (10000, 1) for the training dataset.

We first load the training dataset. The input ``X_train`` is a tuple; the first element is the branch net input, and the second element is the trunk net input.

.. code-block:: python

    d = np.load("antiderivative_unaligned_train.npz", allow_pickle=True)
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]

We also load the testing dataset, and then define the data ``dde.data.Triple``:

.. code-block:: python

    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

Next we define a DeepONet ``dde.nn.DeepONet``. The branch net is chosen as a fully connected neural network of size ``[m, 40, 40]``, and the the trunk net is a fully connected neural network of size ``[dim_x, 40, 40]``.

.. code-block:: python

    m = 100
    dim_x = 1
    net = dde.nn.DeepONet(
        [m, 40, 40],
        [dim_x, 40, 40],
        "relu",
        "Glorot normal",
    )

We define a ``Model``, and then train the model with Adam and learning rate 0.001 for 10000 iterations:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=10000)

Complete code
--------------

.. literalinclude:: ../../../examples/operator/antiderivative_unaligned.py
  :language: python
