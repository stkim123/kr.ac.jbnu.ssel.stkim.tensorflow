from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

# x_data = [[1, 2],
#           [2, 3],
#           [3, 1],
#           [4, 3],
#           [5, 3],
#           [6, 2]]
# y_data = [[0],
#           [0],
#           [0],
#           [1],
#           [1],
#           [1]]


x_data = [[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
          [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
          [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
          [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
          [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
          [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

model = Sequential()
model.add(Dense(1, input_dim=11, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.summary()
model.fit(x_data, y_data, epochs=2000)

weights = model.get_weights()

print("2,1", model.predict_classes(np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]])))
print("6,5", model.predict_classes(np.array([[1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]])))
