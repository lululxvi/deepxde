# DeepXDE

[![Build Status](https://travis-ci.org/lululxvi/deepxde.svg?branch=master)](https://travis-ci.org/lululxvi/deepxde)
[![Documentation Status](https://readthedocs.org/projects/deepxde/badge/?version=latest)](https://deepxde.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/033f58727d674c598558a92da65bf0ed)](https://app.codacy.com/app/lululxvi/deepxde?utm_source=github.com&utm_medium=referral&utm_content=lululxvi/deepxde&utm_campaign=Badge_Grade_Dashboard)
[![PyPI Version](https://badge.fury.io/py/DeepXDE.svg)](https://badge.fury.io/py/DeepXDE)
[![PyPI Downloads](https://pepy.tech/badge/deepxde)](https://pepy.tech/project/deepxde)
[![Conda Version](https://anaconda.org/conda-forge/deepxde/badges/version.svg)](https://anaconda.org/conda-forge/deepxde)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/deepxde.svg)](https://anaconda.org/conda-forge/deepxde)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/lululxvi/deepxde/blob/master/LICENSE)

DeepXDE is a deep learning library on top of [TensorFlow](https://www.tensorflow.org/). Use DeepXDE if you need a deep learning library that

- solves forward and inverse partial differential equations (PDEs) via physics-informed neural network (PINN),
- solves forward and inverse integro-differential equations (IDEs) via PINN,
- solves forward and inverse fractional partial differential equations (fPDEs) via fractional PINN (fPINN),
- approximates functions from multi-fidelity data via multi-fidelity NN (MFNN),
- approximates nonlinear operators via deep operator network (DeepONet),
- approximates functions from a dataset with/without constraints.

**Documentation**: [ReadTheDocs](https://deepxde.readthedocs.io/), [SIAM Rev.](https://doi.org/10.1137/19M1274067), [Slides](https://lululxvi.github.io/files/talks/2020SIAMMDS_MS70.pdf), [Video](https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=13)

**Papers on algorithms**

- Solving PDEs and IDEs via PINN: [SIAM Rev.](https://doi.org/10.1137/19M1274067)
- Solving fPDEs via fPINN: [SIAM J. Sci. Comput.](https://epubs.siam.org/doi/abs/10.1137/18M1229845)
- Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC): [J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999119305340)
- Solving inverse design/topology optimization: [arXiv](https://arxiv.org/abs/2102.04626)
- Learning from multi-fidelity data via MFNN: [PNAS](https://www.pnas.org/content/117/13/7052)
- Learning nonlinear operators via DeepONet: [Nat. Mach. Intell.](https://doi.org/10.1038/s42256-021-00302-5)

## Features

DeepXDE supports

- complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection;
- 6 sampling methods: uniform, pseudorandom, Latin hypercube sampling, Halton sequence, Hammersley sequence, and Sobol sequence;
- multi-physics, i.e., coupled PDEs;
- 5 types of boundary conditions (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC; BCs can be defined on an arbitrary domain or on a point set;
- time-dependent PDEs are solved as easily as time-independent ones by only adding initial conditions;
- residual-based adaptive refinement (RAR);
- uncertainty quantification using dropout;
- two types of neural networks: (stacked/unstacked) fully connected neural network, and residual neural network;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training;
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable. It is easy to customize DeepXDE to meet new demands.

## Installation

DeepXDE requires [TensorFlow](https://www.tensorflow.org/) to be installed. Then, you can install DeepXDE itself.

- Install the stable version with `pip`:

```
$ pip install deepxde
```

- Install the stable version with `conda`:

```
$ conda install -c conda-forge deepxde
```

- For developers, you should clone the folder to your local machine and put it along with your project scripts.

```
$ git clone https://github.com/lululxvi/deepxde.git
```

- Dependencies

  - [Matplotlib](https://matplotlib.org/)
  - [NumPy](http://www.numpy.org/)
  - [scikit-learn](https://scikit-learn.org)
  - [scikit-optimize](https://scikit-optimize.github.io)
  - [SciPy](https://www.scipy.org/)
  - [TensorFlow](https://www.tensorflow.org/)>=2.2.0

## Explore more

- [Demos of forward problems](https://deepxde.readthedocs.io/en/latest/demos/forward.html)
- [Demos of inverse problems](https://deepxde.readthedocs.io/en/latest/demos/inverse.html)
- [More examples](https://github.com/lululxvi/deepxde/tree/master/examples)
- [FAQ](https://deepxde.readthedocs.io/en/latest/user/faq.html)
- [Research papers used DeepXDE](https://deepxde.readthedocs.io/en/latest/user/research.html)
- [API](https://deepxde.readthedocs.io/en/latest/modules/modules.html)

## Cite DeepXDE

If you use DeepXDE for academic research, you are encouraged to cite the following paper:

```
@article{lu2021deepxde,
  author  = {Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  title   = {{DeepXDE}: A deep learning library for solving differential equations},
  journal = {SIAM Review},
  volume  = {63},
  number  = {1},
  pages   = {208-228},
  year    = {2021},
  doi     = {10.1137/19M1274067}
}
```

## Contributing to DeepXDE

First off, thanks for taking the time to contribute!

- **Reporting bugs.** To report a bug, simply open an issue in the GitHub "Issues" section.
- **Suggesting enhancements.** To submit an enhancement suggestion for DeepXDE, including completely new features and minor improvements to existing functionality, let us know by opening an issue.
- **Pull requests.** If you made improvements to DeepXDE, fixed a bug, or had a new example, feel free to send us a pull-request.
- **Asking questions.** To get help on how to use DeepXDE or its functionalities, you can as well open an issue.
- **Answering questions.** If you know the answer to any question in the "Issues", you are welcomed to answer.

## The Team

DeepXDE was originally developed by [Lu Lu](https://lululxvi.github.io/) at the [CRUNCH group](https://www.brown.edu/research/projects/crunch/home) under the supervision of Prof. [George Karniadakis](https://www.brown.edu/research/projects/crunch/george-karniadakis), supported by [PhILMs](https://www.pnnl.gov/computing/philms/).

DeepXDE is currently maintained by [Lu Lu](https://lululxvi.github.io/) with major contributions coming from several talented individuals in various forms and means. A non-exhaustive but growing list needs to mention: Shunyuan Mao, Qi Tang.

## License

[Apache license 2.0](https://github.com/lululxvi/deepxde/blob/master/LICENSE)
