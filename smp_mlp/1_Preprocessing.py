
import os
import numpy as np
from TFANN import MLPR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale

# pth = filePath + 'yahoostock.csv'

dirpath = os.path.dirname(__file__)
filepath = os.path.join(dirpath, 'yahoostock.csv')

A = np.loadtxt(filepath, delimiter=",", skiprows=1, usecols=(1, 4))

#print(len(A[np.where(1500 <= A[:,1])]))
#print(np.where(1500 <= A[:,1]))

A = scale(A)
# y is the dependent variable
y = A[:, 1].reshape(-1, 1)  # y is HIGH
# A contains the independent variable
A = A[:, 0].reshape(-1, 1)  # A is date value
# Plot the high value of the stock price
mpl.plot(A[:, 0], y[:, 0])
mpl.show()