import brainstate as bst
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

from deepxde import pinnx

# PDE
geom = pinnx.geometry.TimeDomain(0, 1).to_dict_point('t')


def pde(x, u_, aux):
    def solve_jac(inp1):
        f1 = lambda i: pinnx.grad.jacobian(lambda inp: net((x[0], inp))['u'][i], inp1, vmap=False)
        return jax.vmap(f1)(np.arange(x[0].shape[0]))

    jacobian = jax.vmap(solve_jac, out_axes=1)(jax.numpy.expand_dims(x[1], 1))
    return u.math.squeeze(jacobian) - aux


# Net
net = bst.nn.Sequential(
    pinnx.nn.DeepONetCartesianProd(
        [50, 128, 128, 128],
        [1, 128, 128, 128],
        "tanh",
        # Hard constraint zero IC
        output_transform=lambda inputs, outputs: outputs * inputs[1].T
    ),
    pinnx.nn.ArrayToDict(u=None)
)

ic = pinnx.icbc.IC(lambda _, aux: {'u': 0})

# Function space
func_space = pinnx.fnspace.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.problem.PDEOperatorCartesianProd(
    geom,
    pde,
    ic,
    func_space,
    eval_pts,
    approximator=net,
    num_function=1000,
    num_fn_test=100,
    batch_size=100,
    num_domain=20,
    num_boundary=2,
    num_test=40
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.0005)).train(iterations=40000)
trainer.saveplot()

v = np.sin(np.pi * eval_pts).T
x = np.linspace(0, 1, num=50)
u = np.ravel(trainer.predict((v, x[:, None]))['u'])
u_true = 1 / np.pi - np.cos(np.pi * x) / np.pi
print(pinnx.metrics.l2_relative_error(u_true, u))
plt.figure()
plt.plot(x, u_true, "k")
plt.plot(x, u, "r")
plt.show()
