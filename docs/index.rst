DeepXDE
===================================

DeepXDE is a deep learning library on top of `TensorFlow <https://www.tensorflow.org/>`_. Use DeepXDE if you need a deep learning library that

- solves forward and inverse partial differential equations (PDEs) via physics-informed neural network (PINN),
- solves forward and inverse integro-differential equations (IDEs) via PINN,
- solves forward and inverse fractional partial differential equations (fPDEs) via fractional PINN (fPINN),
- approximates functions from multi-fidelity data via multi-fidelity NN (MFNN),
- approximates nonlinear operators via deep operator network (DeepONet),
- approximates functions from a dataset with/without constraints.

**Documentation**: `ReadTheDocs <https://deepxde.readthedocs.io/>`_, `Extended abstract <http://ceur-ws.org/Vol-2587/article_14.pdf>`_, `Short paper <https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_2.pdf>`_, `Full paper <https://arxiv.org/abs/1907.04502>`_, `Slides <https://lululxvi.github.io/files/talks/2020AAAI.pdf>`_, `Video <https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=13>`_

**Papers**

- Algorithms & examples

  - Solving PDEs and IDEs via PINN: `Extended abstract <http://ceur-ws.org/Vol-2587/article_14.pdf>`_, `Short paper <https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_2.pdf>`_, `Full paper <https://arxiv.org/abs/1907.04502>`_
  - Solving fPDEs via fPINN: `SIAM J. Sci. Comput. <https://epubs.siam.org/doi/abs/10.1137/18M1229845>`_
  - Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC): `J. Comput. Phys. <https://www.sciencedirect.com/science/article/pii/S0021999119305340>`_
  - Learning from multi-fidelity data via MFNN: `PNAS <https://www.pnas.org/content/117/13/7052>`_, `J. Comput. Phys. <https://www.sciencedirect.com/science/article/pii/S0021999119307260>`_
  - Learning nonlinear operators via DeepONet: `arXiv <https://arxiv.org/abs/1910.03193>`_

- Applications

  - Inverse problems in nano-optics and metamaterials: `Opt. Express <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-28-8-11618>`_

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

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable. It is easy to customize DeepXDE to meet new demands.

User guide
------------

.. toctree::
  :maxdepth: 2

  user/installation
  user/questions_answers
  user/cite_deepxde

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
