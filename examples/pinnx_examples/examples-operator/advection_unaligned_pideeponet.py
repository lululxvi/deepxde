import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from deepxde import pinnx

geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')


# PDE operator
def pde(x, y, aux):
    jacobian = pinnx.grad.jacobian(
        lambda inp: net((inp['v'], pinnx.utils.dict_to_array(inp['u']))),
        {'v': x[0], 'u': {'x': x[1][..., 0], 't': x[1][..., 1]}},
        x='u'
    )
    dy_x = jacobian['y']['u']['x']
    dy_t = jacobian['y']['u']['t']
    return dy_t + dy_x


# Neural network
def periodic(x):
    x, t = x[..., :1], x[..., 1:]
    x = x * 2 * np.pi
    return u.math.concatenate([u.math.cos(x), u.math.sin(x), u.math.cos(2 * x), u.math.sin(2 * x), t], -1)


dim_x = 5
net = bst.nn.Sequential(
    pinnx.nn.DeepONet(
        [50, 128, 128, 128],
        [dim_x, 128, 128, 128],
        "tanh",
        num_outputs=1,
        input_transform=periodic,
    ),
    pinnx.nn.ArrayToDict(y=None),
)

# initial condition
ic = pinnx.icbc.IC(lambda x, aux: {'y': aux})

# Function space
fn_space = pinnx.fnspace.GRF(kernel="ExpSineSquared", length_scale=1)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
problem = pinnx.problem.PDEOperator(
    geomtime,
    pde,
    ic,
    fn_space,
    eval_pts,
    num_function=1000,
    approximator=net,
    function_variables=[0],
    num_fn_test=1000,
    num_domain=250,
    num_initial=50,
    num_test=500
)

trainer = pinnx.Trainer(problem)
trainer.compile(bst.optim.Adam(0.001)).train(iterations=50000)
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
