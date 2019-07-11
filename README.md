# DeepXDE ℒ

[![Build Status](https://travis-ci.org/lululxvi/deepxde.svg?branch=master)](https://travis-ci.org/lululxvi/deepxde)
[![Documentation Status](https://readthedocs.org/projects/deepxde/badge/?version=latest)](https://deepxde.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/033f58727d674c598558a92da65bf0ed)](https://app.codacy.com/app/lululxvi/deepxde?utm_source=github.com&utm_medium=referral&utm_content=lululxvi/deepxde&utm_campaign=Badge_Grade_Dashboard)
[![PyPI Version](https://badge.fury.io/py/DeepXDE.svg)](https://badge.fury.io/py/DeepXDE)
[![Conda Version](https://anaconda.org/conda-forge/deepxde/badges/version.svg)](https://anaconda.org/conda-forge/deepxde)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/lululxvi/deepxde/blob/master/LICENSE)

DeepXDE is a deep learning library for solving differential equations on top of [TensorFlow](https://www.tensorflow.org/).

Use DeepXDE if you need a deep learning library that

- solves partial differential equations (PDEs),
- solves integro-differential equations (IDEs),
- solves fractional partial differential equations (fPDEs),
- solves inverse problems for differential equations,
- approximates functions from a dataset with/without constraints,
- approximates functions from multi-fidelity data.

DeepXDE is extensible to solve other problems in Scientific Machine Learning.

**Documentation**: [ReadTheDocs](https://deepxde.readthedocs.io/)

**DeepXDE Paper**: [arXiv](https://arxiv.org/abs/1907.04502)

## Features

DeepXDE supports

- complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection;
- multi-physics, i.e., coupled PDEs;
- 4 types of boundary conditions: Dirichlet, Neumann, Robin, and periodic;
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

DeepXDE requires [TensorFlow](https://www.tensorflow.org/install/) to be installed.
Then, you can install DeepXDE itself.

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
    - [TensorFlow](https://www.tensorflow.org/)

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

## Why this logo, ℒ?

The art of Scientific Machine Learning with deep learning is to design Loss ℒ.

## License

Apache license 2.0
