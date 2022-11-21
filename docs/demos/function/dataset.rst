Learning a function from a dataset
=======================================

Problem setup
-------------

We will learn a function from a dataset. The dataset used to train the model can be found `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/dataset.train>`_, and the dataset used to test the model can be found `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/dataset.test>`_.


Implementation
--------------

A step by step description of how to implement this code is written below.

Import the DeepXDE library used for this project as described below.

.. code-block:: python

    import deepxde as dde

The next step is to import the dataset needed for the model training.

.. code-block:: python

    fname_train = "../dataset/dataset.train"
    fname_test = "../dataset/dataset.test"

The variables ``fname_train`` and ``fname_test`` are used to import the dataset and recall the dataset later in the code. 

The next step is to define both ``fname_train`` and ``fname_test`` and standardize it in an appropriate form.

.. code-block:: python

    data = dde.data.DataSet(
        fname_train=fname_train,
        fname_test=fname_test,
        col_x=(0,),
        col_y=(1,),
        standardize=True,
    )

After defining the dataset, the specifics of the model are defined. 
The first line defines the layout of the network size used to train the model.
The next line specifies the activation function used ``tanh`` and the initializer as ``Glorot uniform``.

.. code-block:: python

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.nn.FNN(layer_size, activation, initializer)

The model can now be built using ``adam`` as an optimizer with a learning rate of 0.001.
The model is trained with 50000 iterations:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=50000)
    
The best trained model is saved and plotted.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Complete code
-------------

.. literalinclude:: ../../examples/function/dataset.py
  :language: python
