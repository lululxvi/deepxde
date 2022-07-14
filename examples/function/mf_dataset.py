"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde


fname_lo_train = "../dataset/mf_lo_train.dat"
fname_hi_train = "../dataset/mf_hi_train.dat"
fname_hi_test = "../dataset/mf_hi_test.dat"

data = dde.data.MfDataSet(
    fname_lo_train=fname_lo_train,
    fname_hi_train=fname_hi_train,
    fname_hi_test=fname_hi_test,
    col_x=(0,),
    col_y=(1,),
    standardize=True,
)

activation = "tanh"
initializer = "Glorot uniform"
regularization = ["l2", 0.01]
net = dde.nn.MfNN(
    [1] + [20] * 4 + [1],
    [10] * 2 + [1],
    activation,
    initializer,
    regularization=regularization,
)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=80000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
