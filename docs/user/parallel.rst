Parallel training
=================

Introduction
------------

DeepXDE utilizes data-parallel acceleration through the Horovod training framework. To compensate for the memory shortcomings of GPUs, batch size can be increased while employing data-parallel acceleration. 

For :math:`\textrm{size}` GPUs and :math:`\textrm{rank}=1, \cdots, \mathrm{size}`, the data-parallel approach is as follows:
#. We send the same synchronized copy of the neural network to each rank. Each rank evaluates its local loss :math:`\mathcal{L}_\theta^\textrm{rank}` and gradient :math:`\nabla \mathcal{L}_\theta^\textrm{rank}`;
#. The gradients are then averaged using an all-reduce operation, such as the ring all-reduce implemented in Horovod, which is known to be optimal with respect to the number of ranks.

The process is illustrated below for :math:`\textrm{size} = 4`. The ring-allreduce algorithm involves each of the size nodes communicating with two of its peers :math:`2×(size−1)` times.

.. image:: ../images/dataparallel.png
   :align: center
   :width: 400px


Weak and strong scaling
-----------------------
