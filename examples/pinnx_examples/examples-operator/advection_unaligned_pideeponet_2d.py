import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from deepxde import pinnx

# The same problem as advection_unaligned_pideeponet.py
# But consider time as the 2nd space coordinate
# to demonstrate the implementation of 2D problems
geom = pinnx.geometry.Rectangle([0, 0], [1, 1]).to_dict_point('x', 'y')

dim_x = 5


# PDE
def pde(x, y, aux):
    jacobian = pinnx.grad.jacobian(
        lambda inp: net((inp['v'], pinnx.utils.dict_to_array(inp['u']))),
        {'v': x[0], 'u': {'x': x[1][..., 0], 'y': x[1][..., 1]}},
        x='u'
    )
    dy_x = jacobian['y']['u']['x']
    dy_t = jacobian['y']['u']['y']
    return dy_t + dy_x


def func_ic(x, aux):
    return {'y': aux}


def boundary(x, on_boundary):
    return u.math.logical_and(on_boundary, u.math.isclose(x['y'], 0))


ic = pinnx.icbc.DirichletBC(func_ic, boundary)

# Function space
func_space = pinnx.fnspace.GRF(kernel="ExpSineSquared", length_scale=1)


# Net
def periodic(x):
    x, t = x[..., :1], x[..., 1:]
    x = x * 2 * u.math.pi
    return u.math.concatenate(
        [u.math.cos(x),
         u.math.sin(x),
         u.math.cos(2 * x),
         u.math.sin(2 * x), t],
        -1
    )


net = bst.nn.Sequential(
    pinnx.nn.DeepONet(
        [50, 128, 128, 128],
        [dim_x, 128, 128, 128],
        "tanh",
        input_transform=periodic,
    ),
    pinnx.nn.ArrayToDict(y=None),
)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.problem.PDEOperator(
    geom,
    pde,
    ic,
    func_space,
    eval_pts,
    approximator=net,
    num_domain=200,
    num_boundary=200,
    num_function=1000,
    function_variables=[0]
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.0005)).train(iterations=10000)
trainer.saveplot()

x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
u_true = np.sin(2 * np.pi * (x - t[:, None]))
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = np.sin(2 * np.pi * eval_pts)[:, 0]
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = trainer.predict((np.tile(v_branch, (100 * 100, 1)), x_trunk))['y']
u_pred = u_pred.reshape((100, 100))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
print(pinnx.metrics.l2_relative_error(u_true, u_pred))
