DeepXDE
===================================

`DeepXDE <https://github.com/lululxvi/deepxde>`_ is a deep learning library for solving differential equations
on top of `TensorFlow <https://www.tensorflow.org/>`_.

Use DeepXDE if you need a deep learning library that

- solves partial differential equations (PDEs),
- solves integro-differential equations (IDEs),
- solves fractional partial differential equations (fPDEs),
- solves inverse problems for differential equations,
- approximates functions from a dataset with/without constraints,
- approximates functions from multi-fidelity data.

DeepXDE is extensible to solve other problems in scientific computing.

Features
--------

DeepXDE supports

- complex domain geometries without tyranny mesh generation. The basic geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) by operations: union, difference, and intersection;
- multi-physics, i.e., coupled PDEs;
- 4 types of boundary conditions: Dirichlet, Neumann, Robin, and periodic;
- time-dependent PDEs are solved as easily as time-independent ones by only adding initial conditions;
- residue-based adaptive training points;
- uncertainty quantification using dropout;
- four domain geometries: interval, disk, hyercube and hypersphere;
- two types of neural networks: fully connected neural network, and residual neural network;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training;
- compact and nice code, very close to the mathematical formulation.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable.
It is easy to add new functions to each modules to satisfy new requirements.

User guide
------------

.. toctree::
  :maxdepth: 2

  user/installation

API reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  modules/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
