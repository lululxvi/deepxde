import brainstate as bst
import matplotlib.pyplot as plt
import numpy as np

import deepxde
import deepxde.experimental as deepxde_new

# PDE
geom = deepxde_new.geometry.TimeDomain(0, 1).to_dict_point('t')


def pde(x, u, aux):
    jacobian = deepxde_new.grad.jacobian(
        lambda inp: net((inp['v'], inp['t'])),
        {'v': x[0], 't': x[1]},
        x='t'
    )
    return jacobian['u']['t'] - aux


def transform(inputs, outputs):
    return outputs * inputs[1]

# Net
net = bst.nn.Sequential(
    deepxde_new.nn.DeepONet(
        [50, 128, 128, 128],
        [1, 128, 128, 128],
        "tanh",
        # Hard constraint zero IC
        output_transform=transform
    ),
    deepxde_new.nn.ArrayToDict(u=None)
)


ic = deepxde_new.icbc.IC(lambda _, aux: {'u': 0})

# Function space
func_space = deepxde.data.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = deepxde_new.problem.PDEOperator(
    geom,
    pde,
    ic,
    func_space,
    eval_pts,
    approximator=net,
    num_function=1000,
    num_fn_test=1000,
    num_domain=20,
    num_boundary=2,
    num_test=40
)

trainer = deepxde_new.Trainer(data)
trainer.compile(bst.optim.Adam(0.0005)).train(iterations=40000)
trainer.saveplot()

x = np.linspace(0, 1, num=50)
v = np.sin(np.pi * x)
u = np.ravel(trainer.predict((np.tile(v, (50, 1)), x[:, None]))['u'])
u_true = 1 / np.pi - np.cos(np.pi * x) / np.pi
print(deepxde_new.metrics.l2_relative_error(u_true, u))
plt.figure()
plt.plot(x, u_true, "k")
plt.plot(x, u, "r")
plt.show()
