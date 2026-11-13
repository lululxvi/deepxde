import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt

import deepxde.experimental as deepxde


def pde(x, y):
    hessian = net.hessian(x)
    dy_xx = hessian["y"]["x"]["x"]
    return -dy_xx - u.math.pi**2 * u.math.sin(u.math.pi * x["x"])


def func(x):
    return {"y": u.math.sin(u.math.pi * x["x"])}


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN([1] + [50] * 3 + [1], "tanh"),
    deepxde.nn.ArrayToDict(y=None),
)

geom = deepxde.geometry.Interval(-1, 1).to_dict_point("x")
bc = deepxde.icbc.DirichletBC(func)
data = deepxde.problem.PDE(
    geom, pde, bc, net, num_domain=16, num_boundary=2, solution=func, num_test=100
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
trainer.train(iterations=10000)

# Optional: Save the trainer during training.
# checkpointer = experimental.callbacks.ModelCheckpoint(
#     "trainer/trainer", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = experimental.callbacks.MovieDumper(
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
plt.plot(x["x"], y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()
