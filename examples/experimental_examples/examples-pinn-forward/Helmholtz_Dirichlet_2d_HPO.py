import brainstate as bst
import brainunit as u
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import deepxde.experimental as deepxde

# General parameters
d = 2
n = 2
k0 = 2 * np.pi * n
precision_train = 10
precision_test = 30
iterations = 10000


def func(x):
    return {'y': u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])}


def transform(x, y):
    x = deepxde.utils.array_to_dict(x, ["x", "y"], keep_dim=True)
    res = x['x'] * (1 - x['x']) * x['y'] * (1 - x['y'])
    return res * y


def create_model(config):
    def pde(x, y):
        hessian = net.hessian(x)
        dy_xx = hessian['y']['x']['x']
        dy_yy = hessian['y']['y']['y']
        f = (d - 1) * k0 ** 2 * u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])
        return -dy_xx - dy_yy - k0 ** 2 * y['y'] - f

    learning_rate, num_dense_layers, num_dense_nodes, activation = config

    geom = deepxde.geometry.Rectangle([0, 0], [1, 1]).to_dict_point('x', 'y')
    k0 = 2 * np.pi * n
    wave_len = 1 / n

    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    net = deepxde.nn.Model(
        deepxde.nn.DictToArray(x=None, y=None),
        deepxde.nn.FNN(
            [d] + [num_dense_nodes] * num_dense_layers + [1],
            activation,
            bst.init.KaimingUniform(),
            output_transform=transform,
        ),
        deepxde.nn.ArrayToDict(y=None),
    )

    problem = deepxde.problem.PDE(
        geom,
        pde,
        [],
        net,
        num_domain=nx_train ** d,
        num_boundary=2 * d * nx_train,
        solution=func,
        num_test=nx_test ** d,
    )

    trainer = deepxde.Trainer(problem)
    trainer.compile(bst.optim.Adam(learning_rate), metrics=["l2 relative error"])
    return trainer


def train_model(model, config):
    model.train(iterations=iterations)
    loss_test = np.asarray(model.loss_history.loss_test)
    test = loss_test.sum(axis=1).ravel()
    error = test.min()
    return error


# HPO setting
n_calls = 50
dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
]

default_parameters = [1e-3, 4, 50, u.math.sin]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
    global ITERATION

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model(config)
    # possibility to change where we save
    error = train_model(model, config)
    # print(accuracy, 'accuracy is')

    if np.isnan(error):
        error = 10 ** 5

    ITERATION += 1
    return error


ITERATION = 0

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

print(search_result.x)

plot_convergence(search_result)
plot_objective(search_result, show_points=True, size=3.8)
