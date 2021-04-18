Poisson equation in 1D with Dirichlet boundary conditions
=========================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: -\nabla u(x) = \pi^2 \sin(\pi x), \qquad x \in [-1, 1],

with the Dirichlet boundary conditions

.. math:: u(-1)=0, u(1)=0.

The exact solution is :math:`u(x) = \sin(\pi x)`.

Implementation
--------------

Complete code
--------------

.. literalinclude:: ../../examples/Poisson_Dirichlet_1d.py
  :language: python
