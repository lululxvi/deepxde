import brainstate as bst
import brainunit as u

import deepxde.experimental as deepxde


def func(x):
    return {"y": x["x"] * u.math.sin(5 * x["x"])}


layer_size = [1] + [50] * 3 + [1]
net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN(layer_size, "tanh", bst.init.KaimingUniform()),
    deepxde.nn.ArrayToDict(y=None),
)

geom = deepxde.geometry.Interval(-1, 1).to_dict_point("x")
num_train = 100
num_test = 1000
data = deepxde.problem.Function(geom, func, num_train, num_test, approximator=net)

trainer = deepxde.Trainer(data)
uncertainty = deepxde.callbacks.DropoutUncertainty(period=1000)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(
    iterations=30000, callbacks=uncertainty
)
trainer.saveplot(issave=True, isplot=True)
