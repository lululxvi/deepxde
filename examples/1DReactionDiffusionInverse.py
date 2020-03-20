from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import deepxde as dde

#This file uses Mathematica string data as training and uses deepxde for machine learning.



###Read and format Mathematica training data ###########################
# For reading Excel Files
import xlrd
import itertools

file1 =  './1Ddata.xlsx'

# Load in the workbook
wb = xlrd.open_workbook(file1)

#Initialize variables, make sure these match order writtein in Mathematica
t=[]
x=[]
u=[]
dataRaw3 = []

#Extract data from each sheet and rearrage Excel strings to Python List
for i in range( len(wb.sheet_names())):
    sheet = wb.sheet_by_index(i)

    #Need to subtract one to get rid of first row and first column
    rowNum= sheet.nrows
    colNum = sheet.ncols

    dataRaw = [[sheet.cell_value(r, c) for c in range(colNum)] for r in range(rowNum)]
    dataRaw.extend(dataRaw)
    dataRaw2 = list(itertools.chain.from_iterable(dataRaw))
    dataRaw3.append(dataRaw2)


data = list(itertools.chain.from_iterable(dataRaw3))

#Build array for each variable
for j in range(len(data)):
    s = data[j]

	#Convert String to float
    data[j] = float(s)

    #BE CAREFUL!  Change this to match order written in Mathematica
    if j % 3 == 0:
        t.append(data[j])
    if j % 3 == 1:
        x.append(data[j])
    if j % 3 == 2:
        u.append(data[j])

#Formatting issues, turn list to column array

T = np.array(t)
T = T[:, None]

X = np.array(x)
X = X[:, None]

U = np.array(u)
U = U[:, None]


X = np.reshape(X, (-1, 1))
T = np.reshape(T, (-1, 1))

observe_x = np.hstack((X, T))

###Deepxde ########################################################
def main():
    #Initialize Parameters
    D = tf.Variable(1.0)
    rho = tf.Variable(1.0)

	
	#Define diff eq with tensorflow
    def pde(x, y):
        y = y[:, 0:1]
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_t = dy_x[:, 0:1], dy_x[:, 1:]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        return (
            dy_t
            - D * dy_xx
            - rho * y
        )
		
	#Initial Condition, x(t, x) = x(0,x)
    def fun_init(x):
        return 1/(2*np.pi )*np.exp(-.5 * (x[:, 0:1] - 1)**2)

	#Boudary Condition not defined in this case
    #def fun_bc(x):
        #return 1 - x[:, 0:1]

	#Define geometry of region and time 
    geom = dde.geometry.Interval(0, 10)
    timedomain = dde.geometry.TimeDomain(0, 10)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Initial condition for deepxde
    ic1 = dde.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)

    # Define training data for deepxde
    ptset = dde.bc.PointSet(observe_x)
    inside = lambda x, _: ptset.inside(x)
    observe_y1 = dde.DirichletBC(
        geomtime, ptset.values_to_func(U), lambda x, _: ptset.inside(x), component=0
    )

	# Input for deepxde 
    data = dde.data.TimePDE(
        geomtime, #geometry and time domain
        2, #number of inputs?
        pde, # define PDE
        [ic1, observe_y1], #input boudndary conditions, initial conditions, and trainindg data
        num_domain=2000,
        num_boundary=100,
        num_initial=100,
        anchors=observe_x,  
        num_test=50000,
    )

	#define size of neural network, and activation function
    net = dde.maps.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")
	
	#Initiate neural network
    model = dde.Model(data, net)
	
	
	#Run neural network
    model.compile("adam", lr=0.001)
	
	#Extract constants rho and D
    variable = dde.callbacks.VariableValue(
        [D, rho], period=600, filename="variables.dat"
    )

	#Train neural network
    losshistory, train_state = model.train(epochs=100000, callbacks=[variable])
	
	#Display plots after calculation
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)



if __name__ == "__main__":
    main()


