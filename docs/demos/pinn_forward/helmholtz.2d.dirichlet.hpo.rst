Helmholtz equation over a 2D square domain: Hyper-parameter optimization
========================================================================

Finding proper hyper-parameters for PINNs infrastructures is a common issue for practicioners. To remedy this concern, we apply hyper-parameter optimization (HPO) via Gaussian processes (GP)-based Bayesian optimization.

This example is issued from: *Hyper-parameter tuning of physics-informed neural networks: Application to Helmholtz problems*, [`ArXiv preprint <https://arxiv.org/pdf/2205.06704.pdf>`_].

Notice that this script can be easilly adapted to other examples (either forward or inverse problems). 

More scripts are available at `HPOMax <https://github.com/pescap/HPOMax>`_ GitHub repository. 

Problem setup
--------------

We consider the same setting as in `Helmholtz equation over a 2D square domain <https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.dirichlet.html>`_. 

We apply GP-based Bayesian optimization via ``scikit-optimize`` (see `documentation <https://scikit-optimize.github.io/stable/>`_) over 50 calls. We use the `Expected Improvement <https://scikit-optimize.github.io/stable/modules/generated/skopt.acquisition.gaussian_ei.html?highlight=ei#skopt.acquisition.gaussian_ei>`_ as acquisition function, define the minimum test error for each call as the (outer) loss function for the HPO.

We optimize the following hyper-parameters:

- Learning rate :math:`\alpha`;
- Width :math:`N`: number of nodes per layer;
- Depth :math:`L − 1`: number of dense layers;
- Activation function :math:`\sigma`.

We define every configuration as:

.. math:: \lambda := [\alpha, N, L-1, \sigma]

and start with an initial setting :math:`\lambda_0 := [1e-3, 4, 50, \sin]`. 

Implementation
--------------

We highlight the most important parts of the code. At each iteration, the HPO defines a model and trains it. Therefore, we define:

.. code-block:: python

  def create_model(config):
      # Define the model
      return model

which sets the model for a given configuration :math:`\lambda`. Next, we define:      

.. code-block:: python

  def train_model(model, config):
      # Train the model
      # Define the metric we seek to optimize
      return error

which allows to obtain the HPO loss for each configuration. In our case, we seek at minimizing the best test error. We are ready to define the search space and default parameters:

.. code-block:: python   

  dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
  dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
  dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
  dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")

  dimensions = [
      dim_learning_rate,
      dim_num_dense_layers,
      dim_num_dense_nodes,
      dim_activation,
  ]

  default_parameters = [1e-3, 4, 50, "sin"]

Next, we define the ``fitness`` function, which is an input to ``gp_minimize``:

.. code-block:: python   

  @use_named_args(dimensions=dimensions)
  def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):

      config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
      global ITERATION

      print(ITERATION, "it number")
      # Print the hyper-parameters.
      print("learning rate: {0:.1e}".format(learning_rate))
      print("num_dense_layers:", num_dense_layers)
      print("num_dense_nodes:", num_dense_nodes)
      print("activation:", activation)
      print()

      # Create the neural network with these hyper-parameters.
      model = create_model(config)
      # possibility to change where we save
      error = train_model(model, config)
      # print(accuracy, 'accuracy is')

      if np.isnan(error):
          error = 10**5

      ITERATION += 1
      return error

The test error can yield ``nan`` values. We replace this value by a overkill value of ``10**5``. Finally, we apply the GP-based HPO and plot the convergence results:

.. code-block:: python   

  ITERATION = 0

  search_result = gp_minimize(
      func=fitness,
      dimensions=dimensions,
      acq_func="EI",  # Expected Improvement.
      n_calls=n_calls,
      x0=default_parameters,
      random_state=1234,
  )

  search_result.x

  plot_convergence(search_result)
  plot_objective(search_result, show_points=True, size=3.8)


Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Helmholtz_Dirichlet_2d_HPO.py
  :language: python
