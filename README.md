# SciCoNet

SciCoNet (**Sci**entific **Co**mputing Neural **Net**works) is a deep learning library designed for scientific computing on top of [TensorFlow](https://www.tensorflow.org/).

Use SciCoNet if you need a deep learning library that

- approximate functions from a dataset with/without constraints,
- approximate functions from multi-fidelity data,
- solves partial differential equations (PDEs),
- solves integro-differential equations (IDEs),
- solves fractional partial differential equations (fPDEs).

SciCoNet is extensible to solve other problems in scientific computing.

## Features

SciCoNet supports

- uncertainty quantification using dropout; 
- four domain geometries: interval, disk, hyercube and hypersphere;
- two types of neural networks: fully connected neural network, and residual neural network;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training.

SciCoNet is built with four main modules, including

- domain geometry,
- data, i.e., the type of problems and constraints,
- map, i.e., the function space,
- model, which trains the map to match the data and constraints,

and thus is highly-configurable. It is easy to add new functions to each modules to satisfy new requirements.

## Installation

### Dependencies

- [Matplotlib](https://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [SALib](http://salib.github.io/SALib/)
- [scikit-learn](https://scikit-learn.org)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](https://www.tensorflow.org/)

## License

[Apache license 2.0](LICENSE)
