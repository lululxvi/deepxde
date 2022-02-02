# DeepXDE

[![Build Status](https://app.travis-ci.com/lululxvi/deepxde.svg?branch=master)](https://app.travis-ci.com/lululxvi/deepxde)
[![Documentation Status](https://readthedocs.org/projects/deepxde/badge/?version=latest)](https://deepxde.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5c67adbfeabd4ccc9b84d2212c50a342)](https://www.codacy.com/gh/lululxvi/deepxde/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lululxvi/deepxde&amp;utm_campaign=Badge_Grade)
[![PyPI Version](https://badge.fury.io/py/DeepXDE.svg)](https://badge.fury.io/py/DeepXDE)
[![PyPI Downloads](https://pepy.tech/badge/deepxde)](https://pepy.tech/project/deepxde)
[![Conda Version](https://anaconda.org/conda-forge/deepxde/badges/version.svg)](https://anaconda.org/conda-forge/deepxde)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/deepxde.svg)](https://anaconda.org/conda-forge/deepxde)
[![License](https://img.shields.io/github/license/lululxvi/deepxde)](https://github.com/lululxvi/deepxde/blob/master/LICENSE)

DeepXDE is a library for scientific machine learning. Use DeepXDE if you need a deep learning library that

- solves forward and inverse partial differential equations (PDEs) via physics-informed neural network (PINN),
- solves forward and inverse integro-differential equations (IDEs) via PINN,
- solves forward and inverse fractional partial differential equations (fPDEs) via fractional PINN (fPINN),
- approximates nonlinear operators via deep operator network (DeepONet),
- approximates functions from multi-fidelity data via multi-fidelity NN (MFNN),
- approximates functions from a dataset with/without constraints.

DeepXDE supports three tensor libraries as backends: TensorFlow 1.x (`tensorflow.compat.v1` in TensorFlow 2.x), TensorFlow 2.x, and PyTorch. For how to select one, see [Working with different backends](https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends).

**Documentation**: [ReadTheDocs](https://deepxde.readthedocs.io/), [SIAM Rev.](https://doi.org/10.1137/19M1274067), [Slides](https://github.com/lululxvi/tutorials/blob/master/20211210_pinn/pinn.pdf), [Video](https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=13), [Video in Chinese](http://tianyuan.xmu.edu.cn/cn/minicourses/637.html)

**Papers on algorithms**

- Solving PDEs and IDEs via PINN [[SIAM Rev.](https://doi.org/10.1137/19M1274067)], gradient-enhanced PINN (gPINN) [[arXiv](https://arxiv.org/abs/2111.02801)]
- Solving fPDEs via fPINN [[SIAM J. Sci. Comput.](https://epubs.siam.org/doi/abs/10.1137/18M1229845)]
- Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC) [[J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999119305340)]
- Solving inverse design/topology optimization via PINN with hard constraints (hPINN) [[SIAM J. Sci. Comput.](https://doi.org/10.1137/21M1397908)]
- Learning nonlinear operators via DeepONet [[Nat. Mach. Intell.](https://doi.org/10.1038/s42256-021-00302-5), [arXiv](https://arxiv.org/abs/2111.05512)], DeepM&Mnet [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110296), [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110698)]
- Learning from multi-fidelity data via MFNN [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2019.109020), [PNAS](https://www.pnas.org/content/117/13/7052)]

## Features

DeepXDE has implemented many algorithms as shown above and supports many features:

- complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, cuboid, and sphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection.
- multi-physics, i.e., (time-dependent) coupled PDEs.
- 5 types of boundary conditions (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC, which can be defined on an arbitrary domain or on a point set.
- different neural networks, such as (stacked/unstacked) fully connected neural network, residual neural network, and (spatio-temporal) multi-scale fourier feature networks.
- 6 sampling methods: uniform, pseudorandom, Latin hypercube sampling, Halton sequence, Hammersley sequence, and Sobol sequence. The training points can keep the same during training or be resampled every certain iterations.
- conveniently save the model during training, and load a trained model.
- uncertainty quantification using dropout.
- many different (weighted) losses, optimizers, learning rate schedules, metrics, etc.
- callbacks to monitor the internal states and statistics of the model during training, such as early stopping.
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of DeepXDE are loosely coupled, and thus DeepXDE is well-structured and highly configurable. It is easy to customize DeepXDE to meet new demands.

## Installation

DeepXDE requires one of the following backend-specific dependencies to be installed:

- TensorFlow 1.x: [TensorFlow](https://www.tensorflow.org/)>=2.2.0
- TensorFlow 2.x: [TensorFlow](https://www.tensorflow.org/)>=2.2.0 and [TensorFlow Probability](https://www.tensorflow.org/probability)>=0.10.0
- PyTorch: [PyTorch](https://pytorch.org/)

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

- Other dependencies

  - [Matplotlib](https://matplotlib.org/)
  - [NumPy](http://www.numpy.org/)
  - [scikit-learn](https://scikit-learn.org)
  - [scikit-optimize](https://scikit-optimize.github.io)
  - [SciPy](https://www.scipy.org/)

## Explore more

- [Install and Setup](https://deepxde.readthedocs.io/en/latest/user/installation.html)
- [Demos of forward problems](https://deepxde.readthedocs.io/en/latest/demos/forward.html)
- [Demos of inverse problems](https://deepxde.readthedocs.io/en/latest/demos/inverse.html)
- [Demos of function approximation](https://deepxde.readthedocs.io/en/latest/demos/func.html)
- [FAQ](https://deepxde.readthedocs.io/en/latest/user/faq.html)
- [Research papers used DeepXDE](https://deepxde.readthedocs.io/en/latest/user/research.html)
- [API](https://deepxde.readthedocs.io/en/latest/modules/deepxde.html)

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

Also, if you would like your paper to appear [here](https://deepxde.readthedocs.io/en/latest/user/research.html), open an issue in the GitHub "Issues" section.

## Contributing to DeepXDE

First off, thanks for taking the time to contribute!

- **Reporting bugs.** To report a bug, simply open an issue in the GitHub "Issues" section.
- **Suggesting enhancements.** To submit an enhancement suggestion for DeepXDE, including completely new features and minor improvements to existing functionality, let us know by opening an issue.
- **Pull requests.** If you made improvements to DeepXDE, fixed a bug, or had a new example, feel free to send us a pull-request.
- **Asking questions.** To get help on how to use DeepXDE or its functionalities, you can as well open an issue.
- **Answering questions.** If you know the answer to any question in the "Issues", you are welcomed to answer.

**Slack.** The DeepXDE Slack hosts a primary audience of moderate to experienced DeepXDE users and developers for general chat, online discussions, collaboration, etc. If you need a slack invite, please send me an email.

## The Team

DeepXDE was developed by [Lu Lu](https://lu.seas.upenn.edu) under the supervision of Prof. [George Karniadakis](https://www.brown.edu/research/projects/crunch/george-karniadakis) at [Brown University](https://www.brown.edu) from the summer of 2018 to 2020, supported by [PhILMs](https://www.pnnl.gov/computing/philms). DeepXDE was originally self-hosted in Subversion at Brown University, under the name SciCoNet (Scientific Computing Neural Networks). On Feb 7, 2019, SciCoNet was moved from Subversion to GitHub, renamed to DeepXDE.

DeepXDE is currently maintained by [Lu Lu](https://lu.seas.upenn.edu) at [University of Pennsylvania](https://www.upenn.edu) with major contributions coming from several talented individuals in various forms and means. A non-exhaustive but growing list needs to mention: [Shunyuan Mao](https://github.com/smao-astro), [Zongren Zou](https://github.com/ZongrenZou).

## License

[LGPL-2.1 License](https://github.com/lululxvi/deepxde/blob/master/LICENSE)
