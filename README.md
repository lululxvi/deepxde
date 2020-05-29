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

**Documentation**: [ReadTheDocs](https://deepxde.readthedocs.io/), [Extended abstract](http://ceur-ws.org/Vol-2587/article_14.pdf), [Short paper](https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_2.pdf), [Full paper](https://arxiv.org/abs/1907.04502), [Slides](https://lululxvi.github.io/files/talks/2020AAAI.pdf), [Video](https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=13)

**Papers**

- Algorithms & examples

  - Solving PDEs and IDEs via PINN: [Extended abstract](http://ceur-ws.org/Vol-2587/article_14.pdf), [Short paper](https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_2.pdf), [Full paper](https://arxiv.org/abs/1907.04502)
  - Solving fPDEs via fPINN: [SIAM J. Sci. Comput.](https://epubs.siam.org/doi/abs/10.1137/18M1229845)
  - Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC): [J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999119305340)
  - Learning from multi-fidelity data via MFNN: [PNAS](https://www.pnas.org/content/117/13/7052), [J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999119307260)
  - Learning nonlinear operators via DeepONet: [arXiv](https://arxiv.org/abs/1910.03193)

- Applications

  - Inverse problems in nano-optics and metamaterials: [Opt. Express](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-28-8-11618)

## Features

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

## Installation

DeepXDE requires [TensorFlow](https://www.tensorflow.org/) to be installed. Both TensorFlow 1 and TensorFlow 2 can be used as the DeepXDE backend, but TensorFlow 1 is recommended, because in my tests TensorFlow 2 is 2x~3x slower than TensorFlow 1.

Then, you can install DeepXDE itself. If you use Python 2, you need to install DeepXDE using `pip`.

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
  - [SALib](http://salib.github.io/SALib/)
  - [scikit-learn](https://scikit-learn.org)
  - [SciPy](https://www.scipy.org/)
  - [TensorFlow](https://www.tensorflow.org/)>=1.14.0

## Explore more

- [Examples](https://github.com/lululxvi/deepxde/tree/master/examples)
- [Questions & Answers](https://deepxde.readthedocs.io/en/latest/user/questions_answers.html)

## Cite DeepXDE

If you use DeepXDE for academic research, you are encouraged to cite the following paper:

```
@article{lu2019deepxde,
  author  = {Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George E.},
  title   = {{DeepXDE}: A deep learning library for solving differential equations},
  journal = {arXiv preprint arXiv:1907.04502},
  year    = {2019}
}
```

## Contributing to DeepXDE

First off, thanks for taking the time to contribute!

- **Reporting bugs.** To report a bug, simply open an issue in the GitHub "Issues" section.
- **Suggesting enhancements.** To submit an enhancement suggestion for DeepXDE, including completely new features and minor improvements to existing functionality, let us know by opening an issue.
- **Pull requests.** If you made improvements to DeepXDE, fixed a bug, or had a new example, feel free to send us a pull-request.
- **Questions.** To get help on how to use DeepXDE or its functionalities, you can as well open an issue.

## License

[Apache license 2.0](https://github.com/lululxvi/deepxde/blob/master/LICENSE)
