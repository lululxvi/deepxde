Poisson equation in 1D with Dirichlet boundary conditions
=========================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: -\Delta u = \pi^2 \sin(\pi x), \qquad x \in [-1, 1],

with the Dirichlet boundary conditions

.. math:: u(-1)=0, \quad u(1)=0.

The exact solution is :math:`u(x) = \sin(\pi x)`.

Implementation
--------------
1)We first define the ``Geometry`` 

.. code-block:: python

  geom = dde.geometry.Interval(-1, 1)
      
1)Define the ``pde residual`` function of Poisson equation

.. code-block:: python

  def pde(x, y):
      dy_xx = dde.grad.hessian(y, x)
      return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)
        
2)Define ``boundary conditions`` and ``Data`` which is geometry + pde + BC + training points using DeepXDE inbuilt functions as shown

.. code-block:: python
  
  bc = dde.DirichletBC(geom, func, boundary)
  data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)    
    
3)We use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50 

.. code-block:: python

  layer_size = [1] + [50] * 3 + [1]
  activation = "tanh"
  initializer = "Glorot uniform"
  net = dde.maps.FNN(layer_size, activation, initializer)
     
4)Bulid the Model and Compile using the ``MODEL`` and ``Complile`` functions of DeepXDE as shown

.. code-block:: python

  model = dde.Model(data, net)
  model.``compile("adam", lr=0.001, metrics=["l2 relative error"])
  checkpointer = dde.callbacks.ModelCheckpoint("model/model.ckpt", verbose=1, save_better_only=True)
  movie = dde.callbacks.MovieDumper("model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func)
  

5)Train the model using ``Train`` function of deepXDE, include the callbacks as shown   

.. code-block:: python

  losshistory, train_state = model.train(epochs=10000, callbacks=[checkpointer, movie])
  
6)Predict values by using the ``Predict`` function of deepXDE 

.. code-block:: python 

   y = model.predict(x, operator=pde)

Complete code
--------------

.. literalinclude:: ../../examples/Poisson_Dirichlet_1d.py
  :language: python
