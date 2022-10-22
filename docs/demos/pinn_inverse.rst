Demos of Inverse Problems
=========================

Here are some demos of solving inverse problems of PDEs.

ODEs
----

.. toctree::
   :maxdepth: 1

   pinn_inverse/lorenz.inverse
   pinn_inverse/lorenz.inverse.forced

Time-independent PDEs
---------------------

.. toctree::
   :maxdepth: 1

   pinn_inverse/elliptic.inverse.field

- `Inverse problem for the Poisson equation with unknown forcing field with training points resampling <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field_batch.py>`_
- `Inverse problem for Brinkman-Forchheimer model <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/brinkman_forchheimer.py>`_
- `Inverse problem for the diffusion-reaction system with a space-dependent reaction rate <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_reaction_rate.py>`_

Time-dependent PDEs
-------------------

.. toctree::
   :maxdepth: 1

   pinn_inverse/diffusion.1d.inverse
   pinn_inverse/reaction.inverse
   
- `Inverse problem for the Navier-Stokes equation of incompressible flow around cylinder <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/Navier_Stokes_inverse.py>`_

fractional PDEs
---------------

- `Inverse problem for the fractional Poisson equation in 1D <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_1d_inverse.py>`_
- `Inverse problem for the fractional Poisson equation in 2D <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_2d_inverse.py>`_
