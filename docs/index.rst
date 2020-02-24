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
- approximates functions from multi-fidelity data,
- approximates operators.

DeepXDE is extensible to solve other problems in Scientific Machine Learning.

**Documentation**: `ReadTheDocs <https://deepxde.readthedocs.io/>`_, `short paper <https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_2.pdf>`_, `full paper <https://arxiv.org/abs/1907.04502>`_, `slides <https://lululxvi.github.io/files/talks/2020PIML.pdf>`_

**Papers**

- Algorithms & examples

  - Solving PDEs and IDEs: `short paper <https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_2.pdf>`_, `full paper <https://arxiv.org/abs/1907.04502>`_, `slides <https://lululxvi.github.io/files/talks/2020PIML.pdf>`_
  - Solving fPDEs: `SIAM J. Sci. Comput. <https://epubs.siam.org/doi/abs/10.1137/18M1229845>`_
  - Solving stochastic PDEs: `J. Comput. Phys. <https://www.sciencedirect.com/science/article/pii/S0021999119305340>`_
  - Multi-fidelity neural network: `arXiv <https://arxiv.org/abs/1903.00104>`_
  - DeepONet to learn nonlinear operators: `arXiv <https://arxiv.org/abs/1910.03193>`_

- Applications

  - Inverse problems in nano-optics and metamaterials: `arXiv <https://arxiv.org/abs/1912.01085>`_

Features
--------

DeepXDE supports

- complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection;
- multi-physics, i.e., coupled PDEs;
- 5 types of boundary conditions (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC;
- time-dependent PDEs are solved as easily as time-independent ones by only adding initial conditions;
- residual-based adaptive refinement (RAR);
- uncertainty quantification using dropout;
- two types of neural networks: fully connected neural network, and residual neural network;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training;
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable.
It is easy to customize DeepXDE to meet new demands.

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
