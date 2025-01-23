# Copyright DeepXDE Limited. All Rights Reserved.
#
# Licensed under the GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import brainstate as bst
import brainunit as u

from deepxde import pinnx


def func(x):
    return {'y': x['x'] * u.math.sin(5 * x['x'])}


layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN(layer_size, "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(y=None),
)

geom = pinnx.geometry.Interval(-1, 1).to_dict_point('x')
num_train = 100
num_test = 1000
data = pinnx.problem.Function(
    geom, func, num_train, num_test,
    approximator=net
)

trainer = pinnx.Trainer(data)
uncertainty = pinnx.callbacks.DropoutUncertainty(period=1000)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=30000, callbacks=uncertainty)
trainer.saveplot(issave=True, isplot=True)
