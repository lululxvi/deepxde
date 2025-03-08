import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt

from deepxde import pinnx


def pde(x, y):
    hessian = net.hessian(x)
    dy_xx = hessian["y"]["x"]["x"]
    return -dy_xx - u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'])


def func(x):
    return {'y': u.math.sin(u.math.pi * x['x'])}


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [50] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

geom = pinnx.geometry.Interval(-1, 1).to_dict_point('x')
bc = pinnx.icbc.DirichletBC(func)
data = pinnx.problem.PDE(
    geom, pde, bc, net, num_domain=16, num_boundary=2, solution=func, num_test=100
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
trainer.train(iterations=10000)

# Optional: Save the trainer during training.
# checkpointer = pinnx.callbacks.ModelCheckpoint(
#     "trainer/trainer", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = pinnx.callbacks.MovieDumper(
#     "trainer/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# trainer.train(iterations=10000, callbacks=[checkpointer, movie])

trainer.saveplot(issave=True, isplot=True)

# Optional: Restore the saved trainer with the smallest training loss
# trainer.restore(f"trainer/trainer-{train_state.best_step}.ckpt", verbose=1)
# Plot PDE residual
x = geom.uniform_points(1000, True)
y = pde(x, trainer.predict(x))
plt.figure()
plt.plot(x['x'], y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()
