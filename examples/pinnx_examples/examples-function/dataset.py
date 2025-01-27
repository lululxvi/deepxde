import os

import brainstate as bst
import numpy as np

from deepxde import pinnx

PATH = os.path.dirname(os.path.abspath(__file__))
train_data = np.loadtxt(os.path.join(PATH, '..', 'dataset', 'dataset.train'))
test_data = np.loadtxt(os.path.join(PATH, '..', 'dataset', 'dataset.test'))

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [50] * 3 + [1], "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(y=None),
)

data = pinnx.problem.DataSet(
    X_train={'x': train_data[:, 0]},
    y_train={'y': train_data[:, 1]},
    X_test={'x': test_data[:, 0]},
    y_test={'y': test_data[:, 1]},
    standardize=True,
    approximator=net,
)

model = pinnx.Trainer(data)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=50000)
model.saveplot(issave=True, isplot=True)
