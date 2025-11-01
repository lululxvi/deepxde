import brainstate as bst
import brainunit as u

import deepxde.experimental as deepxde


def func(x):
    return {"y": x["x"] * u.math.sin(5 * x["x"])}


geom = deepxde.geometry.Interval(-1, 1).to_dict_point("x")
num_train = 160
num_test = 100

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN([1] + [20] * 3 + [1], "tanh", bst.init.LecunUniform()),
    deepxde.nn.ArrayToDict(y=None),
)

data = deepxde.problem.Function(geom, func, num_train, num_test, approximator=net)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(
    iterations=10000
)
trainer.saveplot(issave=False, isplot=True)
