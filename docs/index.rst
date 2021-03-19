DeepXDE
===================================

DeepXDE is a deep learning library on top of `TensorFlow <https://www.tensorflow.org/>`_. Use DeepXDE if you need a deep learning library that

- solves forward and inverse partial differential equations (PDEs) via physics-informed neural network (PINN),
- solves forward and inverse integro-differential equations (IDEs) via PINN,
- solves forward and inverse fractional partial differential equations (fPDEs) via fractional PINN (fPINN),
- approximates functions from multi-fidelity data via multi-fidelity NN (MFNN),
- approximates nonlinear operators via deep operator network (DeepONet),
- approximates functions from a dataset with/without constraints.

**Documentation**: `ReadTheDocs <https://deepxde.readthedocs.io/>`_, `SIAM Rev. <https://doi.org/10.1137/19M1274067>`_, `Slides <https://lululxvi.github.io/files/talks/2020SIAMMDS_MS70.pdf>`_, `Video <https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=13>`_

**Papers on algorithms**

- Solving PDEs and IDEs via PINN: `SIAM Rev. <https://doi.org/10.1137/19M1274067>`_
- Solving fPDEs via fPINN: `SIAM J. Sci. Comput. <https://epubs.siam.org/doi/abs/10.1137/18M1229845>`_
- Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC): `J. Comput. Phys. <https://www.sciencedirect.com/science/article/pii/S0021999119305340>`_
- Solving inverse design/topology optimization: `arXiv <https://arxiv.org/abs/2102.04626>`_
- Learning from multi-fidelity data via MFNN: `PNAS <https://www.pnas.org/content/117/13/7052>`_
- Learning nonlinear operators via DeepONet: `Nat. Mach. Intell. <https://doi.org/10.1038/s42256-021-00302-5>`_

Features
--------

DeepXDE supports

- complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection;
- multi-physics, i.e., coupled PDEs;
- 5 types of boundary conditions (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC;
- time-dependent PDEs are solved as easily as time-independent ones by only adding initial conditions;
- residual-based adaptive refinement (RAR);
- uncertainty quantification using dropout;
- two types of neural networks: (stacked/unstacked) fully connected neural network, and residual neural network;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training;
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable. It is easy to customize DeepXDE to meet new demands.

User guide
------------

.. toctree::
  :maxdepth: 2

  user/installation
  user/faq
  user/research
  user/cite_deepxde
  user/team

API reference
-------------

If you are looking for information on a specific function, class or method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  modules/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
