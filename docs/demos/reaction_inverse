Inverse problem for DIffusion-Reaction Systems with Dirichlet boundary conditions
=================================================================================

Problem setup
--------------

We will solve the Diffusion reaction system in porous media, the equations are:

.. math:: \frac{\partial C_A}{\partial t} = D\frac{\partial^2 C_A}{\partial x^2} -{k_f}{C_A}{{C_B}^2}
.. math:: \frac{\partial C_B}{\partial t} = D\frac{\partial^2 C_B}{\partial x^2} -2{k_f}{C_A}{{C_B}^2}, \qquad x \in [0, 1], \qquad t \in [0, 10]

with the Dirichlet boundary conditions

.. math:: C_A(x,0)=C_B(x,0)=e^{-20x}, \quad C_A(0,t)=C_B(0,t)=1, \quad C_A(1,t)=C_B(1,t)=0.

The reference solution is <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/reaction.npz>

Implementation
--------------

We use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20.

Complete code
--------------

.. literalinclude:: ../../examples/reaction_inverse.py
  :language: python
