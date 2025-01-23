# Copyright DeepXDE Limited. All Rights Reserved.
#
# Licensed under the GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import brainstate as bst
import brainunit as u
import jax.tree
import numpy as np

from deepxde import pinnx

unit_of_space = u.meter
unit_of_speed = u.meter / u.second
unit_of_t = u.second
unit_of_pressure = u.pascal

spatial_domain = pinnx.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
temporal_domain = pinnx.geometry.TimeDomain(0, 1)
spatio_temporal_domain = pinnx.geometry.GeometryXTime(spatial_domain, temporal_domain)
spatio_temporal_domain = spatio_temporal_domain.to_dict_point(
    x=unit_of_space,
    y=unit_of_space,
    z=unit_of_space,
    t=unit_of_t,
)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=unit_of_space,
                         y=unit_of_space,
                         z=unit_of_space,
                         t=unit_of_t),
    pinnx.nn.FNN([4] + 4 * [50] + [4], "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(u_vel=unit_of_speed,
                         v_vel=unit_of_speed,
                         w_vel=unit_of_speed,
                         p=unit_of_pressure),
)

a = 1
d = 1
Re = 1
rho = 1 * u.kilogram / u.meter ** 3
mu = 1 * u.pascal * u.second


@bst.compile.jit
def pde(x, u):
    jacobian = net.jacobian(x)
    x_hessian = net.hessian(x, y=['u_vel', 'v_vel', 'w_vel'], xi=['x'], xj=['x'])
    y_hessian = net.hessian(x, y=['u_vel', 'v_vel', 'w_vel'], xi=['y'], xj=['y'])
    z_hessian = net.hessian(x, y=['u_vel', 'v_vel', 'w_vel'], xi=['z'], xj=['z'])

    u_vel, v_vel, w_vel, p = u['u_vel'], u['v_vel'], u['w_vel'], u['p']

    du_vel_dx = jacobian['u_vel']['x']
    du_vel_dy = jacobian['u_vel']['y']
    du_vel_dz = jacobian['u_vel']['z']
    du_vel_dt = jacobian['u_vel']['t']
    du_vel_dx_dx = x_hessian['u_vel']['x']['x']
    du_vel_dy_dy = y_hessian['u_vel']['y']['y']
    du_vel_dz_dz = z_hessian['u_vel']['z']['z']

    dv_vel_dx = jacobian['v_vel']['x']
    dv_vel_dy = jacobian['v_vel']['y']
    dv_vel_dz = jacobian['v_vel']['z']
    dv_vel_dt = jacobian['v_vel']['t']
    dv_vel_dx_dx = x_hessian['v_vel']['x']['x']
    dv_vel_dy_dy = y_hessian['v_vel']['y']['y']
    dv_vel_dz_dz = z_hessian['v_vel']['z']['z']

    dw_vel_dx = jacobian['w_vel']['x']
    dw_vel_dy = jacobian['w_vel']['y']
    dw_vel_dz = jacobian['w_vel']['z']
    dw_vel_dt = jacobian['w_vel']['t']
    dw_vel_dx_dx = x_hessian['w_vel']['x']['x']
    dw_vel_dy_dy = y_hessian['w_vel']['y']['y']
    dw_vel_dz_dz = z_hessian['w_vel']['z']['z']

    dp_dx = jacobian['p']['x']
    dp_dy = jacobian['p']['y']
    dp_dz = jacobian['p']['z']

    momentum_x = (
        rho * (du_vel_dt + (u_vel * du_vel_dx + v_vel * du_vel_dy + w_vel * du_vel_dz))
        + dp_dx - mu * (du_vel_dx_dx + du_vel_dy_dy + du_vel_dz_dz)
    )
    momentum_y = (
        rho * (dv_vel_dt + (u_vel * dv_vel_dx + v_vel * dv_vel_dy + w_vel * dv_vel_dz))
        + dp_dy - mu * (dv_vel_dx_dx + dv_vel_dy_dy + dv_vel_dz_dz)
    )
    momentum_z = (
        rho * (dw_vel_dt + (u_vel * dw_vel_dx + v_vel * dw_vel_dy + w_vel * dw_vel_dz))
        + dp_dz - mu * (dw_vel_dx_dx + dw_vel_dy_dy + dw_vel_dz_dz)
    )
    continuity = du_vel_dx + dv_vel_dy + dw_vel_dz

    return [momentum_x, momentum_y, momentum_z, continuity]


@bst.compile.jit(static_argnums=1)
def icbc_cond_func(x, include_p: bool = False):
    x = {k: v.mantissa for k, v in x.items()}

    u_ = (
        -a
        * (u.math.exp(a * x['x']) * u.math.sin(a * x['y'] + d * x['z'])
           + u.math.exp(a * x['z']) * u.math.cos(a * x['x'] + d * x['y']))
        * u.math.exp(-(d ** 2) * x['t'])
    )
    v = (
        -a
        * (u.math.exp(a * x['y']) * u.math.sin(a * x['z'] + d * x['x'])
           + u.math.exp(a * x['x']) * u.math.cos(a * x['y'] + d * x['z']))
        * u.math.exp(-(d ** 2) * x['t'])
    )
    w = (
        -a
        * (u.math.exp(a * x['z']) * u.math.sin(a * x['x'] + d * x['y'])
           + u.math.exp(a * x['y']) * u.math.cos(a * x['z'] + d * x['x']))
        * u.math.exp(-(d ** 2) * x['t'])
    )
    p = (
        -0.5
        * a ** 2
        * (
            u.math.exp(2 * a * x['x'])
            + u.math.exp(2 * a * x['y'])
            + u.math.exp(2 * a * x['z'])
            + 2
            * u.math.sin(a * x['x'] + d * x['y'])
            * u.math.cos(a * x['z'] + d * x['x'])
            * u.math.exp(a * (x['y'] + x['z']))
            + 2
            * u.math.sin(a * x['y'] + d * x['z'])
            * u.math.cos(a * x['x'] + d * x['y'])
            * u.math.exp(a * (x['z'] + x['x']))
            + 2
            * u.math.sin(a * x['z'] + d * x['x'])
            * u.math.cos(a * x['y'] + d * x['z'])
            * u.math.exp(a * (x['x'] + x['y']))
        )
        * u.math.exp(-2 * d ** 2 * x['t'])
    )

    r = {
        'u_vel': u_ * unit_of_speed,
        'v_vel': v * unit_of_speed,
        'w_vel': w * unit_of_speed
    }
    if include_p:
        r['p'] = p * unit_of_pressure
    return r


bc = pinnx.icbc.DirichletBC(icbc_cond_func)
ic = pinnx.icbc.IC(icbc_cond_func)

problem = pinnx.problem.TimePDE(
    spatio_temporal_domain,
    pde,
    [bc, ic],
    net,
    num_domain=50000,
    num_boundary=5000,
    num_initial=5000,
    num_test=10000,
)

model = pinnx.Trainer(problem)

model.compile(bst.optim.Adam(1e-3)).train(iterations=30000)
model.compile(bst.optim.LBFGS(1e-3)).train(5000, display_every=200)

x, y, z = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
t_0 = np.zeros(1000)
t_1 = np.ones(1000)
X_0 = dict(
    x=np.ravel(x) * unit_of_space,
    y=np.ravel(y) * unit_of_space,
    z=np.ravel(z) * unit_of_space,
    t=t_0 * unit_of_t
)
X_1 = dict(
    x=np.ravel(x) * unit_of_space,
    y=np.ravel(y) * unit_of_space,
    z=np.ravel(z) * unit_of_space,
    t=t_1 * unit_of_t
)
output_0 = model.predict(X_0)
output_1 = model.predict(X_1)

out_exact_0 = icbc_cond_func(X_0, True)
out_exact_1 = icbc_cond_func(X_1, True)

f_0 = pde(X_0, output_0)
f_1 = pde(X_1, output_1)
residual_0 = jax.tree.map(lambda x: np.mean(np.absolute(x)), f_0)
residual_1 = jax.tree.map(lambda x: np.mean(np.absolute(x)), f_1)

print("Accuracy at t = 0:")
print("Mean residual:", residual_0)
print("L2 relative error:", pinnx.metrics.l2_relative_error(output_0, out_exact_0))
print("\n")
print("Accuracy at t = 1:")
print("Mean residual:", residual_1)
print("L2 relative error:", pinnx.metrics.l2_relative_error(output_1, out_exact_1))
