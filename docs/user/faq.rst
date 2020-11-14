FAQ
===========

If you have any questions about DeepXDE, first read the papers/slides and watch the video at the `DeepXDE homepage <https://deepxde.readthedocs.io>`_ and also check the following list of frequently asked DeepXDE questions. To get further help, you can open an issue in the GitHub "Issues" section.

- | **Q**: DeepXDE failed to run.
  | **A**: `#2`_, `#3`_, `#5`_
- | **Q**: What is the expected output of DeepXDE? How can I visualize the results?
  | **A**: `#4`_, `#9`_, `#17`_, `#48`_, `#53`_, `#73`_, `#77`_
- | **Q**: How can I define a new geometry?
  | **A**: `#32`_, `#38`_
- | **Q**: How can I implement new ODEs/PDEs?
  | **A**: `#12`_, `#13`_, `#21`_, `#22`_, `#74`_, `#78`_, `#79`_, `#124`_
- | **Q**: How can I implement new IDEs?
  | **A**: `#95`_
- | **Q**: More details and examples about initial conditions.
  | **A**: `#19`_, `#75`_, `#104`_, `#134`_
- | **Q**: More details and examples about boundary conditions.
  | **A**: `#6`_, `#10`_, `#15`_, `#16`_, `#22`_, `#26`_, `#33`_, `#38`_, `#40`_, `#44`_, `#49`_, `#115`_, `#140`_
- | **Q**: By default, initial/boundary conditions are enforced in DeepXDE as soft constraints. How can I enforce them as hard constraints?
  | **A**: `#36`_, `#90`_, `#92`_
- | **Q**: Define an inverse problem to solve unknown parameters/fields in the PDEs or initial/boundary conditions.
  | **A**: `#55`_, `#76`_, `#86`_, `#114`_, `#120`_, `#125`_
- | **Q**: How does DeepXDE choose the training points? How can I use some specific training points?
  | **A**: `#32`_, `#57`_, `#64`_
- | **Q**: How can I give different weights to different residual points?
  | **A**: `#45`_
- | **Q**: I failed to train the network or get the right solution, e.g., the training loss is large.
  | **A**: `#15`_, `#22`_, `#33`_, `#41`_, `#61`_, `#62`_, `#80`_, `#84`_, `#85`_, `#108`_, `#126`_, `#141`_
- | **Q**: How can I use a trained model for new predictions?
  | **A**: `#10`_, `#18`_, `#93`_
- | **Q**: How can I save a trained model and then load the model later?
  | **A**: `#54`_, `#57`_, `#58`_, `#63`_, `#103`_
- | **Q**: Residual-based adaptive refinement (RAR).
  | **A**: `#63`_
- | **Q**: By default, DeepXDE uses ``float32``. How can I use ``float64``?
  | **A**: `#28`_
- | **Q**: More details about DeepXDE source code, and want to modify DeepXDE, e.g., to use multiple GPUs and mini batch.
  | **A**: `#35`_, `#39`_, `#66`_, `#68`_, `#69`_, `#91`_, `#99`_, `#131`_
- | **Q**: Examples collected from users.
  | **A**: `Lotka–Volterra <https://github.com/lululxvi/deepxde/issues/85>`_, `Potential flow around a cylinder <https://github.com/lululxvi/deepxde/issues/49>`_, `Laminar Incompressible flow passing a step <https://github.com/lululxvi/deepxde/issues/80>`_
- | **Q**: Questions about multi-fidelity neutral networks.
  | **A**: `#94`_

.. _#2: https://github.com/lululxvi/deepxde/issues/2
.. _#3: https://github.com/lululxvi/deepxde/issues/3
.. _#4: https://github.com/lululxvi/deepxde/issues/4
.. _#5: https://github.com/lululxvi/deepxde/issues/5
.. _#6: https://github.com/lululxvi/deepxde/issues/6
.. _#9: https://github.com/lululxvi/deepxde/issues/9
.. _#10: https://github.com/lululxvi/deepxde/issues/10
.. _#12: https://github.com/lululxvi/deepxde/issues/12
.. _#13: https://github.com/lululxvi/deepxde/issues/13
.. _#15: https://github.com/lululxvi/deepxde/issues/15
.. _#16: https://github.com/lululxvi/deepxde/issues/16
.. _#17: https://github.com/lululxvi/deepxde/issues/17
.. _#18: https://github.com/lululxvi/deepxde/issues/18
.. _#19: https://github.com/lululxvi/deepxde/issues/19
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
