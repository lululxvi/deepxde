DeepXDE
=======

`DeepXDE <https://github.com/lululxvi/deepxde>`_ is a library for scientific machine learning and physics-informed learning. Use DeepXDE if you need a deep learning library that

- solves forward and inverse ordinary/partial differential equations (ODEs/PDEs) via physics-informed neural network (PINN),
- solves forward and inverse integro-differential equations (IDEs) via PINN,
- solves forward and inverse fractional partial differential equations (fPDEs) via fractional PINN (fPINN),
- approximates operators via deep operator network (DeepONet, MIONet, DeepM&Mnet),
- approximates functions from multi-fidelity data via multi-fidelity NN (MFNN).

DeepXDE supports four tensor libraries as backends: TensorFlow 1.x (``tensorflow.compat.v1`` in TensorFlow 2.x), TensorFlow 2.x, PyTorch, and JAX (in development). For how to select one, see `Working with different backends <https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends>`_.

**Documentation**: `ReadTheDocs <https://deepxde.readthedocs.io/>`_, `SIAM Rev. <https://doi.org/10.1137/19M1274067>`_, `Slides <https://github.com/lululxvi/tutorials/blob/master/20211210_pinn/pinn.pdf>`_, `Video <https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=13>`_, `Video in Chinese <http://tianyuan.xmu.edu.cn/cn/minicourses/637.html>`_

**Papers on algorithms**

- Solving ODEs, PDEs and IDEs via PINN [`SIAM Rev. <https://doi.org/10.1137/19M1274067>`_], gradient-enhanced PINN (gPINN) [`arXiv <https://arxiv.org/abs/2111.02801>`_]
- Solving fPDEs via fPINN [`SIAM J. Sci. Comput. <https://epubs.siam.org/doi/abs/10.1137/18M1229845>`_]
- Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC) [`J. Comput. Phys. <https://www.sciencedirect.com/science/article/pii/S0021999119305340>`_]
- Solving inverse design/topology optimization via PINN with hard constraints (hPINN) [`SIAM J. Sci. Comput. <https://doi.org/10.1137/21M1397908>`_]
- Learning operators via DeepONet [`Nat. Mach. Intell. <https://doi.org/10.1038/s42256-021-00302-5>`_, `arXiv <https://arxiv.org/abs/2111.05512>`_], MIONet [`arXiv <https://arxiv.org/abs/2202.06137>`_], DeepM&Mnet [`J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2021.110296>`_, `J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2021.110698>`_]
- Learning from multi-fidelity data via MFNN [`J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2019.109020>`_, `PNAS <https://www.pnas.org/content/117/13/7052>`_]

.. image:: images/pinn.png

.. image:: images/deeponet.png

.. image:: images/backend.png

Features
--------

DeepXDE has implemented many algorithms as shown above and supports many features:

- enables the user code to be compact, resembling closely the mathematical formulation.
- **complex domain geometries** without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, sphere, hypercube, and hypersphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection.
- 5 types of **boundary conditions** (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC, which can be defined on an arbitrary domain or on a point set.
- different **neural networks**: fully connected neural network (FNN), stacked FNN, residual neural network, (spatio-temporal) multi-scale fourier feature networks, etc.
- 6 **sampling methods**: uniform, pseudorandom, Latin hypercube sampling, Halton sequence, Hammersley sequence, and Sobol sequence. The training points can keep the same during training or be resampled every certain iterations.
- different **optimizers**: Adam, L-BFGS, etc.
- conveniently **save** the model during training, and **load** a trained model.
- **callbacks** to monitor the internal states and statistics of the model during training: early stopping, etc.
- **uncertainty quantification** using dropout.
- **float32** and **float64**.
- many other useful features: different (weighted) losses, learning rate schedules, metrics, etc.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable. It is easy to customize DeepXDE to meet new demands.

User guide
----------

.. toctree::
  :maxdepth: 2

  user/installation

.. toctree::
  :maxdepth: 1

  demos/pinn_forward
  demos/pinn_inverse
  demos/function
  user/faq

.. toctree::
  :maxdepth: 2

  user/research
  user/cite_deepxde
  user/team

API reference
-------------

If you are looking for information on a specific function, class or method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2
  :caption: API

  modules/deepxde
  modules/deepxde.data
  modules/deepxde.geometry
  modules/deepxde.icbc
  modules/deepxde.nn
  modules/deepxde.nn.jax
  modules/deepxde.nn.pytorch
  modules/deepxde.nn.tensorflow
  modules/deepxde.nn.tensorflow_compat_v1
  modules/deepxde.optimizers
  modules/deepxde.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
