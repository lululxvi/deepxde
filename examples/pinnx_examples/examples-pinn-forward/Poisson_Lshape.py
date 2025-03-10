import brainstate as bst

from deepxde import pinnx


def pde(x, y):
    hessian = net.hessian(x)
    dy_xx = hessian["u"]["x"]["x"]
    dy_yy = hessian["u"]["y"]["y"]
    return -dy_xx - dy_yy - 1


geom = pinnx.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
geom = geom.to_dict_point("x", "y")
bc = pinnx.icbc.DirichletBC(lambda x: {'u': 0})

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None, y=None),
    pinnx.nn.FNN([2] + [50] * 4 + [1], "tanh"),
    pinnx.nn.ArrayToDict(u=None),
)

data = pinnx.problem.PDE(geom, pde, bc, net, num_domain=1200, num_boundary=120, num_test=1500)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(1e-3)).train(iterations=50000)
trainer.compile(bst.optim.LBFGS(1e-3)).train(10000)
trainer.saveplot(issave=True, isplot=True)
