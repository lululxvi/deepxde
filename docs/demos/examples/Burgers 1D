Burgers equation in 1D with Dirichlet boundary conditions
=========================================================

Problem setup
--------------

We will solve a Burgers equation:

.. math:: \frac{du}{dt} + u\frac{du}{dx} = \nu\frac{du}{dx^2}, \qquad x \in [-1, 1], \qquad t \in [0, 1]

with the Dirichlet boundary conditions

.. math:: u(-1,t)=u(1,t)=0, \quad u(x) = - \sin(\pi x).

The reference solution is <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz>

Implementation
--------------

We use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20.

Complete code
--------------

.. literalinclude:: ../../examples/Burgers.py
  :language: python
