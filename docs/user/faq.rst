FAQ
===

If you have any questions about DeepXDE, first read the papers/slides and watch the video at the `DeepXDE homepage <https://deepxde.readthedocs.io>`_ and also check the following list of frequently asked DeepXDE questions. To get further help, you can open a discussion in the GitHub `Discussions <https://github.com/lululxvi/deepxde/discussions>`_.

General usage
-------------

- | **Q**: DeepXDE failed to run.
  | **A**: `#5`_
- | **Q**: By default, DeepXDE uses ``float32``. How can I use ``float64``?
  | **A**: `#28`_
- | **Q**: I want to set the global random seeds.
  | **A**: `#353`_
- | **Q**: GPU.
  | **A**: `#464`_
- | **Q**: How can I use a trained model for new predictions?
  | **A**: `#669`_, `#676`_, `#93`_, `#177`_
- | **Q**: How can I save a trained model and then load the model later?
  | **A**: `#54`_, `#57`_, `#58`_, `#63`_, `#103`_, `#206`_, `#254`_, `#468`_
- | **Q**: More details about DeepXDE source code, and want to modify DeepXDE.
  | **A**: `#68`_, `#346`_

PINN
----

- | **Q**: What is the output of DeepXDE? How can I visualize the results?
  | **A**: `#668`_, `#9`_, `#675`_, `#48`_, `#53`_, `#73`_, `#77`_, `#171`_, `#217`_, `#218`_, `#223`_, `#274`_, `#276`_
- | **Q**: More details and examples about geometry.
  | **A**: `#32`_, `#38`_, `#161`_, `#264`_, `#278`_, `#332`_
- | **Q**: How can I implement new ODEs/PDEs, e.g., compute derivatives, complicated PDEs, complicated coefficients?
  | **A**: `#670`_, `#21`_, `#22`_, `#74`_, `#78`_, `#79`_, `#124`_, `#172`_, `#185`_, `#193`_, `#194`_, `#246`_, `#302`_, `#377`_, `#421`_, `#451`_, `#465`_, `#478`_, `#480`_
- | **Q**: More details and examples about initial conditions.
  | **A**: `#677`_, `#75`_, `#104`_, `#134`_, `#398`_, `#485`_
- | **Q**: More details and examples about boundary conditions.
  | **A**: `#6`_, `#669`_, `#673`_, `#674`_, `#22`_, `#26`_, `#33`_, `#38`_, `#40`_, `#44`_, `#49`_, `#115`_, `#140`_, `#156`_, `#352`_, `#365`_, `#457`_, `#475`_
- | **Q**: By default, initial/boundary conditions are enforced in DeepXDE as soft constraints. How can I enforce them as hard constraints?
  | **A**: `#36`_, `#90`_, `#92`_, `#252`_
- | **Q**: How can I use a dataset of the solution?
  | **A**: `#362`_, `#487`_
- | **Q**: I failed to train the network or get the right solution, e.g., large training loss, unbalanced losses.
  | **A**: `#673`_, `#22`_, `#33`_, `#41`_, `#61`_, `#62`_, `#80`_, `#84`_, `#85`_, `#108`_, `#126`_, `#141`_, `#188`_, `#247`_, `#305`_, `#321`_
- | **Q**: L-BFGS.
  | **A**: `#364`_, `#395`_
- | **Q**: Implement certain features for the input, such as Fourier features.
  | **A**: `#277`_, `#477`_
- | **Q**: A standard network doesn't work for me, and I want to implement a network with a special structure/property.
  | **A**: `#355`_, `#363`_, `#466`_
- | **Q**: Implement new losses or constraints.
  | **A**: `#174`_, `#286`_, `#311`_, `#356`_, `#460`_, `#467`_, `#504`_
- | **Q**: How can I implement new IDEs?
  | **A**: `#95`_, `#198`_
- | **Q**: Solve PDEs with complex numbers.
  | **A**: `#284`_
- | **Q**: Solve inverse problems with unknown parameters/fields in the PDEs or initial/boundary conditions.
  | **A**: `#55`_, `#76`_, `#86`_, `#114`_, `#120`_, `#125`_, `#178`_, `#208`_, `#235`_, `#380`_
- | **Q**: Solve parametric PDEs.
  | **A**: `#273`_, `#299`_, `#379`_, `#482`_
- | **Q**: How does DeepXDE choose the training points? How can I use some specific training points?
  | **A**: `#32`_, `#57`_, `#64`_, `#345`_
- | **Q**: How can I give different weights to different residual points?
  | **A**: `#45`_
- | **Q**: Residual-based adaptive refinement (RAR).
  | **A**: `#63`_
- | **Q**: I want to customize network training/optimization, e.g., mini-batch.
  | **A**: `#166`_, `#307`_, `#320`_, `#331`_
- | **Q**: I want to get more information about the network, such as the values and gradients of hidden layers.
  | **A**: `#149`_, `#490`_
- | **Q**: More details about DeepXDE source code, or I want to modify DeepXDE.
  | **A**: `#35`_, `#39`_, `#66`_, `#69`_, `#91`_, `#99`_, `#131`_, `#163`_, `#175`_, `#202`_, `#357`_
- | **Q**: Examples collected from users.
  | **A**: `Lotka–Volterra <https://github.com/lululxvi/deepxde/issues/85>`_, `Potential flow around a cylinder <https://github.com/lululxvi/deepxde/issues/49>`_, `Laminar Incompressible flow passing a step <https://github.com/lululxvi/deepxde/issues/80>`_, `Shallow water equations <https://github.com/lululxvi/deepxde/issues/247>`_, `2D time-dependent heat equation <https://github.com/lululxvi/deepxde/issues/61>`_

DeepONet
--------

- | **Q**:
  | **A**:

Multi-fidelity NN
-----------------

- | **Q**: How is MFNN trained?
  | **A**: `#94`_, `#450`_
- | **Q**: Transfer learning of MFNN.
  | **A**: `#195`_
- | **Q**: How can I use a trained model for new predictions?
  | **A**: `#324`_

.. _#668: https://github.com/lululxvi/deepxde/discussions/668
.. _#5: https://github.com/lululxvi/deepxde/issues/5
.. _#6: https://github.com/lululxvi/deepxde/issues/6
.. _#9: https://github.com/lululxvi/deepxde/issues/9
.. _#669: https://github.com/lululxvi/deepxde/discussions/669
.. _#670: https://github.com/lululxvi/deepxde/discussions/670
.. _#673: https://github.com/lululxvi/deepxde/discussions/673
.. _#674: https://github.com/lululxvi/deepxde/discussions/674
.. _#675: https://github.com/lululxvi/deepxde/discussions/675
.. _#676: https://github.com/lululxvi/deepxde/discussions/676
.. _#677: https://github.com/lululxvi/deepxde/discussions/677
.. _#21: https://github.com/lululxvi/deepxde/issues/21
.. _#22: https://github.com/lululxvi/deepxde/issues/22
.. _#26: https://github.com/lululxvi/deepxde/issues/26
.. _#28: https://github.com/lululxvi/deepxde/issues/28
.. _#32: https://github.com/lululxvi/deepxde/issues/32
.. _#33: https://github.com/lululxvi/deepxde/issues/33
.. _#35: https://github.com/lululxvi/deepxde/issues/35
.. _#36: https://github.com/lululxvi/deepxde/issues/36
.. _#38: https://github.com/lululxvi/deepxde/issues/38
.. _#39: https://github.com/lululxvi/deepxde/issues/39
.. _#40: https://github.com/lululxvi/deepxde/issues/40
.. _#41: https://github.com/lululxvi/deepxde/issues/41
.. _#44: https://github.com/lululxvi/deepxde/issues/44
.. _#45: https://github.com/lululxvi/deepxde/issues/45
.. _#48: https://github.com/lululxvi/deepxde/issues/48
.. _#49: https://github.com/lululxvi/deepxde/issues/49
.. _#53: https://github.com/lululxvi/deepxde/issues/53
.. _#54: https://github.com/lululxvi/deepxde/issues/54
.. _#55: https://github.com/lululxvi/deepxde/issues/55
.. _#57: https://github.com/lululxvi/deepxde/issues/57
.. _#58: https://github.com/lululxvi/deepxde/issues/58
.. _#61: https://github.com/lululxvi/deepxde/issues/61
.. _#62: https://github.com/lululxvi/deepxde/issues/62
.. _#63: https://github.com/lululxvi/deepxde/issues/63
.. _#64: https://github.com/lululxvi/deepxde/issues/64
.. _#66: https://github.com/lululxvi/deepxde/issues/66
.. _#68: https://github.com/lululxvi/deepxde/issues/68
.. _#69: https://github.com/lululxvi/deepxde/issues/69
.. _#73: https://github.com/lululxvi/deepxde/issues/73
.. _#74: https://github.com/lululxvi/deepxde/issues/74
.. _#75: https://github.com/lululxvi/deepxde/issues/75
.. _#76: https://github.com/lululxvi/deepxde/issues/76
.. _#77: https://github.com/lululxvi/deepxde/issues/77
.. _#78: https://github.com/lululxvi/deepxde/issues/78
.. _#79: https://github.com/lululxvi/deepxde/issues/79
.. _#80: https://github.com/lululxvi/deepxde/issues/80
.. _#84: https://github.com/lululxvi/deepxde/issues/84
.. _#85: https://github.com/lululxvi/deepxde/issues/85
.. _#86: https://github.com/lululxvi/deepxde/issues/86
.. _#90: https://github.com/lululxvi/deepxde/issues/90
.. _#91: https://github.com/lululxvi/deepxde/issues/91
.. _#92: https://github.com/lululxvi/deepxde/issues/92
.. _#93: https://github.com/lululxvi/deepxde/issues/93
.. _#94: https://github.com/lululxvi/deepxde/issues/94
.. _#95: https://github.com/lululxvi/deepxde/issues/95
.. _#99: https://github.com/lululxvi/deepxde/issues/99
.. _#103: https://github.com/lululxvi/deepxde/issues/103
.. _#104: https://github.com/lululxvi/deepxde/issues/104
.. _#108: https://github.com/lululxvi/deepxde/issues/108
.. _#114: https://github.com/lululxvi/deepxde/issues/114
.. _#115: https://github.com/lululxvi/deepxde/issues/115
.. _#120: https://github.com/lululxvi/deepxde/issues/120
.. _#124: https://github.com/lululxvi/deepxde/issues/124
.. _#125: https://github.com/lululxvi/deepxde/issues/125
.. _#126: https://github.com/lululxvi/deepxde/issues/126
.. _#131: https://github.com/lululxvi/deepxde/issues/131
.. _#134: https://github.com/lululxvi/deepxde/issues/134
.. _#140: https://github.com/lululxvi/deepxde/issues/140
.. _#141: https://github.com/lululxvi/deepxde/issues/141
.. _#149: https://github.com/lululxvi/deepxde/issues/149
.. _#156: https://github.com/lululxvi/deepxde/issues/156
.. _#161: https://github.com/lululxvi/deepxde/issues/161
.. _#163: https://github.com/lululxvi/deepxde/issues/163
.. _#166: https://github.com/lululxvi/deepxde/issues/166
.. _#171: https://github.com/lululxvi/deepxde/issues/171
.. _#172: https://github.com/lululxvi/deepxde/issues/172
.. _#174: https://github.com/lululxvi/deepxde/issues/174
.. _#175: https://github.com/lululxvi/deepxde/issues/175
.. _#177: https://github.com/lululxvi/deepxde/issues/177
.. _#178: https://github.com/lululxvi/deepxde/issues/178
.. _#185: https://github.com/lululxvi/deepxde/issues/185
.. _#188: https://github.com/lululxvi/deepxde/issues/188
.. _#193: https://github.com/lululxvi/deepxde/issues/193
.. _#194: https://github.com/lululxvi/deepxde/issues/194
.. _#195: https://github.com/lululxvi/deepxde/issues/195
.. _#198: https://github.com/lululxvi/deepxde/issues/198
.. _#202: https://github.com/lululxvi/deepxde/issues/202
.. _#206: https://github.com/lululxvi/deepxde/issues/206
.. _#208: https://github.com/lululxvi/deepxde/issues/208
.. _#217: https://github.com/lululxvi/deepxde/issues/217
.. _#218: https://github.com/lululxvi/deepxde/issues/218
.. _#223: https://github.com/lululxvi/deepxde/issues/223
.. _#235: https://github.com/lululxvi/deepxde/issues/235
.. _#246: https://github.com/lululxvi/deepxde/issues/246
.. _#247: https://github.com/lululxvi/deepxde/issues/247
.. _#252: https://github.com/lululxvi/deepxde/issues/252
.. _#254: https://github.com/lululxvi/deepxde/issues/254
.. _#264: https://github.com/lululxvi/deepxde/issues/264
.. _#273: https://github.com/lululxvi/deepxde/issues/273
.. _#274: https://github.com/lululxvi/deepxde/issues/274
.. _#276: https://github.com/lululxvi/deepxde/issues/276
.. _#277: https://github.com/lululxvi/deepxde/issues/277
.. _#278: https://github.com/lululxvi/deepxde/issues/278
.. _#284: https://github.com/lululxvi/deepxde/issues/284
.. _#286: https://github.com/lululxvi/deepxde/issues/286
.. _#299: https://github.com/lululxvi/deepxde/issues/299
.. _#302: https://github.com/lululxvi/deepxde/issues/302
.. _#305: https://github.com/lululxvi/deepxde/issues/305
.. _#307: https://github.com/lululxvi/deepxde/issues/307
.. _#311: https://github.com/lululxvi/deepxde/issues/311
.. _#320: https://github.com/lululxvi/deepxde/issues/320
.. _#321: https://github.com/lululxvi/deepxde/issues/321
.. _#324: https://github.com/lululxvi/deepxde/issues/324
.. _#331: https://github.com/lululxvi/deepxde/issues/331
.. _#332: https://github.com/lululxvi/deepxde/issues/332
.. _#345: https://github.com/lululxvi/deepxde/issues/345
.. _#346: https://github.com/lululxvi/deepxde/issues/346
.. _#352: https://github.com/lululxvi/deepxde/issues/352
.. _#353: https://github.com/lululxvi/deepxde/issues/353
.. _#355: https://github.com/lululxvi/deepxde/issues/355
.. _#356: https://github.com/lululxvi/deepxde/issues/356
.. _#357: https://github.com/lululxvi/deepxde/issues/357
.. _#362: https://github.com/lululxvi/deepxde/issues/362
.. _#363: https://github.com/lululxvi/deepxde/issues/363
.. _#364: https://github.com/lululxvi/deepxde/issues/364
.. _#365: https://github.com/lululxvi/deepxde/issues/365
.. _#377: https://github.com/lululxvi/deepxde/issues/377
.. _#379: https://github.com/lululxvi/deepxde/issues/379
.. _#380: https://github.com/lululxvi/deepxde/issues/380
.. _#395: https://github.com/lululxvi/deepxde/issues/395
.. _#398: https://github.com/lululxvi/deepxde/issues/398
.. _#421: https://github.com/lululxvi/deepxde/issues/421
.. _#450: https://github.com/lululxvi/deepxde/issues/450
.. _#451: https://github.com/lululxvi/deepxde/issues/451
.. _#457: https://github.com/lululxvi/deepxde/issues/457
.. _#460: https://github.com/lululxvi/deepxde/issues/460
.. _#464: https://github.com/lululxvi/deepxde/issues/464
.. _#465: https://github.com/lululxvi/deepxde/issues/465
.. _#466: https://github.com/lululxvi/deepxde/issues/466
.. _#467: https://github.com/lululxvi/deepxde/issues/467
.. _#468: https://github.com/lululxvi/deepxde/issues/468
.. _#475: https://github.com/lululxvi/deepxde/issues/475
.. _#477: https://github.com/lululxvi/deepxde/issues/477
.. _#478: https://github.com/lululxvi/deepxde/issues/478
.. _#480: https://github.com/lululxvi/deepxde/issues/480
.. _#482: https://github.com/lululxvi/deepxde/issues/482
.. _#485: https://github.com/lululxvi/deepxde/issues/485
.. _#487: https://github.com/lululxvi/deepxde/issues/487
.. _#490: https://github.com/lululxvi/deepxde/issues/490
.. _#504: https://github.com/lululxvi/deepxde/issues/504
