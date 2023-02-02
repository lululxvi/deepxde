Demos of Forward Problems
=========================

Here are some demos of solving forward problems of PDEs.

ODEs
----

.. toctree::
   :maxdepth: 1

   pinn_forward/ode.system
   pinn_forward/lotka.volterra
   pinn_forward/ode.2nd

Time-independent PDEs
---------------------

.. toctree::
   :maxdepth: 1

   pinn_forward/poisson.1d.dirichlet
   pinn_forward/poisson.1d.neumanndirichlet
   pinn_forward/poisson.1d.dirichletrobin
   pinn_forward/poisson.1d.dirichletperiodic
   pinn_forward/poisson.1d.pointsetoperator
   pinn_forward/poisson.dirichlet.1d.exactbc
   pinn_forward/poisson.1d.multiscaleFourier
   pinn_forward/poisson.Lshape
   pinn_forward/laplace.disk
   pinn_forward/eulerbeam
   pinn_forward/helmholtz.2d.dirichlet
   pinn_forward/helmholtz.2d.dirichlet.hpo
   pinn_forward/helmholtz.2d.neumann.hole
   pinn_forward/helmholtz.2d.sound.hard.abc

Time-dependent PDEs
-------------------

.. toctree::
   :maxdepth: 1

   pinn_forward/burgers
   pinn_forward/heat
   pinn_forward/heat.resample
   pinn_forward/diffusion.1d
   pinn_forward/diffusion.1d.exactBC
   pinn_forward/diffusion.1d.resample
   pinn_forward/diffusion.reaction
   pinn_forward/burgers.rar
   pinn_forward/allen.cahn
   pinn_forward/klein.gordon
   pinn_forward/Kovasznay.flow

- `Beltrami flow <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py>`_
- `Wave propagation with spatio-temporal multi-scale Fourier feature architecture <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py>`_
- `Schrodinger equation <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Schrodinger.ipynb>`_

Integro-differential equations
------------------------------

- `Integro-differential equation <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ide.py>`_
- `Volterra IDE <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py>`_

fractional PDEs
---------------

- `fractional Poisson equation in 1D <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_1d.py>`_
- `fractional Poisson equation in 2D <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_2d.py>`_
- `fractional Poisson equation in 3D <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_3d.py>`_
- `fractional diffusion equation <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_diffusion_1d.py>`_
