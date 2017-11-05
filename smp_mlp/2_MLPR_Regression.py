
from smp_mlp import TFANN
import os
import numpy as np
from TFANN import MLPR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale


dirpath = os.path.dirname(__file__)
filepath = os.path.join(dirpath, 'yahoostock.csv')

A = np.loadtxt(filepath, delimiter=",", skiprows=1, usecols=(1, 4))

A = scale(A)
# y is the dependent variable
y = A[:, 1].reshape(-1, 1)  # y is HIGH
# A contains the independent variable
A = A[:, 0].reshape(-1, 1)  # A is date value
# Plot the high value of the stock price

#Number of neurons in the input layer
i = 1
#Number of neurons in the output layer
o = 1
#Number of neurons in the hidden layers
h = 32
#The list of layer sizes
layers = [i, h, h, h, h, h, h, h, h, h, o]

# use tanh for activation function
# learnRate=1e-3,
# loss function loss='l2' --> --> YH, Y: tf.squared_difference(Y, YH)
# maxIter=1024
mlpr = TFANN.MLPR(layers, maxIter = 1000, tol = 0.40, reg = 0.1, verbose = True) # reg = 0.1 looks better
# mlpr = TFANN.MLPR(layers, maxIter = 1000, tol = 0.40, reg = 0.001, verbose = True)
# mlpr = TFANN.MLPR(layers, maxIter = 1000, tol = 0.40, reg = 0.00001, verbose = True)
# mlpr = TFANN.MLPR(layers, maxIter = 1000, tol = 0.40, reg = None, verbose = True) #to be worse


#Length of the hold-out period
nDays = 5
n = len(A)
#Learn the data
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])

#Begin prediction
yHat = mlpr.predict(A)
#Plot the results
mpl.plot(A, y, c='#b0403f')
mpl.plot(A, yHat, c='#5aa9ab')
mpl.show()