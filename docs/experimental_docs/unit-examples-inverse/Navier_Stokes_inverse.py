"""
An inverse problem of the Navier-Stokes equation of incompressible flow around cylinder with Re=100

References: https://doi.org/10.1016/j.jcp.2018.10.045 Section 4.1.1
"""

import re

import brainstate as bst
import jax.tree
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import brainunit as u
import deepxde.experimental as deepxde

unit_of_x = u.meter
unit_of_y = u.meter
unit_of_t = u.second
unit_of_rho = u.kilogram / u.meter**3
unit_of_c1 = u.UNITLESS
unit_of_c2 = u.meter2 / u.second
unit_of_u = u.meter / u.second
unit_of_v = u.meter / u.second
unit_of_p = u.pascal

# true values
C1true = 1.0
C2true = 0.01 * unit_of_c2


# Load training data
def load_training_data(num):
    data = loadmat("../dataset/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Problem
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0]
    y_train = data_domain[idx, 1]
    t_train = data_domain[idx, 2]
    u_train = data_domain[idx, 3]
    v_train = data_domain[idx, 4]
    p_train = data_domain[idx, 5]
    # return [x_train, y_train, t_train, u_train, v_train, p_train]
    return [x_train * unit_of_x, y_train * unit_of_y, t_train * unit_of_t,
            u_train * unit_of_u, v_train * unit_of_v, p_train * unit_of_p]


# Parameters to be identified
C1 = bst.ParamState(0.0)
C2 = bst.ParamState(0.0 * unit_of_c2)


# Define Navier Stokes Equations (Time-dependent PDEs)
def Navier_Stokes_Equation(x, y):
    jacobian = net.jacobian(x)
    x_hessian = net.hessian(x, y=['u', 'v'], xi=['x'], xj=['x'])
    y_hessian = net.hessian(x, y=['u', 'v'], xi=['y'], xj=['y'])

    u = y['u']
    v = y['v']
    p = y['p']
    du_x = jacobian['u']['x']
    du_y = jacobian['u']['y']
    du_t = jacobian['u']['t']
    dv_x = jacobian['v']['x']
    dv_y = jacobian['v']['y']
    dv_t = jacobian['v']['t']
    dp_x = jacobian['p']['x']
    dp_y = jacobian['p']['y']
    du_xx = x_hessian['u']['x']['x']
    du_yy = y_hessian['u']['y']['y']
    dv_xx = x_hessian['v']['x']['x']
    dv_yy = y_hessian['v']['y']['y']
    continuity = du_x + dv_y
    x_momentum = unit_of_rho * du_t + unit_of_rho * C1.value * (u * du_x + v * du_y) + dp_x - unit_of_rho * C2.value * (du_xx + du_yy)
    y_momentum = unit_of_rho * dv_t + unit_of_rho * C1.value * (u * dv_x + v * dv_y) + dp_y - unit_of_rho * C2.value * (dv_xx + dv_yy)
    return [continuity, x_momentum, y_momentum]


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=unit_of_x, y=unit_of_y, t=unit_of_t),
    deepxde.nn.FNN([3] + [50] * 6 + [3], "tanh"),
    deepxde.nn.ArrayToDict(u=unit_of_u, v=unit_of_v, p=unit_of_p),
)

# Define Spatio-temporal domain
# Rectangular
Lx_min, Lx_max = 1.0, 8.0
Ly_min, Ly_max = -2.0, 2.0
# Spatial domain: X × Y = [1, 8] × [−2, 2]
space_domain = deepxde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
# Time domain: T = [0, 7]
time_domain = deepxde.geometry.TimeDomain(0, 7)
# Spatio-temporal domain
geomtime = deepxde.geometry.GeometryXTime(space_domain, time_domain)
geomtime = geomtime.to_dict_point(x=unit_of_x, y=unit_of_y, t=unit_of_t)

# Get the training data: num = 7000
[ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=7000)
ob_xyt = {"x": ob_x, "y": ob_y, "t": ob_t}
ob_yv = {"u": ob_u, "v": ob_v, }
observe_bc = deepxde.icbc.PointSetBC(ob_xyt, ob_yv)

# Training datasets and Loss
problem = deepxde.problem.TimePDE(
    geomtime,
    Navier_Stokes_Equation,
    [observe_bc],
    net,
    num_domain=700,
    num_boundary=200,
    num_initial=100,
    anchors=ob_xyt,
)

# Neural Network setup
model = deepxde.Trainer(problem, external_trainable_variables=[C1, C2])

# callbacks for storing results
fnamevar = "variables.dat"
variable = deepxde.callbacks.VariableValue([C1, C2], period=100, filename=fnamevar)

# Compile, train and save trainer
model.compile(bst.optim.Adam(1e-3)).train(iterations=10000, callbacks=[variable],
                                          display_every=1000, disregard_previous_best=True)
model.saveplot(issave=True, isplot=True)
model.compile(bst.optim.Adam(1e-4)).train(iterations=10000, callbacks=[variable],
                                          display_every=1000, disregard_previous_best=True)
model.saveplot(issave=True, isplot=True)

# trainer.save(save_path = "./NS_inverse_model/trainer")
f = model.predict(ob_xyt, operator=Navier_Stokes_Equation)
print("Mean residual:", jax.tree.map(lambda x: u.math.mean(u.math.abs(x)), f))

# Plot Variables:
# reopen saved data using callbacks in fnamevar
lines = open(fnamevar, "r").readlines()
# read output data in fnamevar
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
l, c = Chat.shape
plt.semilogy(range(0, l * 100, 100), Chat[:, 0], "r-")
plt.semilogy(range(0, l * 100, 100), Chat[:, 1], "k-")
plt.semilogy(range(0, l * 100, 100), np.ones(Chat[:, 0].shape) * C1true, "r--")
plt.semilogy(range(0, l * 100, 100), np.ones(Chat[:, 1].shape) * C2true, "k--")
plt.legend(["C1hat", "C2hat", "True C1", "True C2"], loc="right")
plt.xlabel("Epochs")
plt.title("Variables")
plt.show()

# Plot the velocity distribution of the flow field:
for t in range(0, 8):
    t = t * unit_of_t
    [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=140000)

    xyt_pred = {"x": ob_x, "y": ob_y, "t": t * np.ones((len(ob_x),))}
    uvp_pred = model.predict(xyt_pred)

    x_pred, y_pred, t_pred = xyt_pred['x'], xyt_pred['y'], xyt_pred['t']
    u_pred, v_pred, p_pred = uvp_pred['u'], uvp_pred['v'], uvp_pred['p']
    x_true = ob_x[ob_t == t]
    y_true = ob_y[ob_t == t]
    u_true = ob_u[ob_t == t]
    fig, ax = plt.subplots(2, 1)
    cntr0 = ax[0].tricontourf(x_pred.mantissa, y_pred.mantissa, u_pred.mantissa, levels=80, cmap="rainbow")
    cb0 = plt.colorbar(cntr0, ax=ax[0])
    cntr1 = ax[1].tricontourf(x_true.mantissa, y_true.mantissa, u_true.mantissa, levels=80, cmap="rainbow")
    cb1 = plt.colorbar(cntr1, ax=ax[1])
    ax[0].set_title("u-PINN " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[0].set_ylabel("Y", fontsize=7.5, family="Arial")
    ax[1].set_title("u-Reference solution " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[1].set_ylabel("Y", fontsize=7.5, family="Arial")
    fig.tight_layout()
    plt.show()
