import brainstate as bst
import numpy as np

from deepxde import pinnx


def gen_traindata():
    data = np.load("../dataset/Lorenz.npz")
    return data["t"], data["y"]


C1 = bst.ParamState(8.0)
C2 = bst.ParamState(20.0)
C3 = bst.ParamState(-3.0)


def Lorenz_system(x, y):
    """
    Lorenz system.

    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (15 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3
    """
    jacobian = net.jacobian(x)
    y1, y2, y3 = y['y1'], y['y2'], y['y3']
    dy1_x = jacobian['y1']['t']
    dy2_x = jacobian['y2']['t']
    dy3_x = jacobian['y3']['t']
    return [
        dy1_x - C1.value * (y2 - y1),
        dy2_x - y1 * (C2.value - y3) + y2,
        dy3_x - y1 * y2 + C3.value * y3,
    ]


geom = pinnx.geometry.TimeDomain(0, 3).to_dict_point('t')

# Initial conditions
ic = pinnx.icbc.IC(lambda x: {'y1': -8, 'y2': 7, 'y3': 27})

# Get the train data
observe_t, ob_y = gen_traindata()
observe_t = {'t': observe_t.flatten()}
observe_bc = pinnx.icbc.PointSetBC(
    observe_t,
    {'y1': ob_y[:, 0], 'y2': ob_y[:, 1], 'y3': ob_y[:, 2]}
)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(t=None),
    pinnx.nn.FNN([1] + [40] * 3 + [3], "tanh"),
    pinnx.nn.ArrayToDict(y1=None, y2=None, y3=None),
)

data = pinnx.problem.PDE(
    geom,
    Lorenz_system,
    [ic, observe_bc],
    net,
    num_domain=400,
    num_boundary=20,
    anchors=observe_t,
)

variable = pinnx.callbacks.VariableValue([C1, C2, C3], period=600, filename="../../examples/pinn_inverse/variables.dat")

trainer = pinnx.Trainer(data, external_trainable_variables=[C1, C2, C3])
trainer.compile(bst.optim.Adam(0.001)).train(iterations=50000, callbacks=[variable])
# trainer.compile(bst.optim.LBFGS(1e-3)).train(10000, callbacks=[variable])
trainer.saveplot(issave=True, isplot=True)
