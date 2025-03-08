import brainstate as bst
import brainunit as u

from deepxde import pinnx


def func(x):
    return {'y': x['x'] * u.math.sin(5 * x['x'])}


geom = pinnx.geometry.Interval(-1, 1).to_dict_point('x')
num_train = 160
num_test = 100

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh", bst.init.LecunUniform()),
    pinnx.nn.ArrayToDict(y=None),
)

data = pinnx.problem.Function(
    geom, func, num_train, num_test,
    approximator=net
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=False, isplot=True)
