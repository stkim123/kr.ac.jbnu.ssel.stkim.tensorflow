import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# predict a temperature with y(x) = w0 + w1x + w2x^2 + w3x^3 + w4x^4
def predict(x):
    result = 0.0
    for n in range(0, 5):
        result += w_val[n][0] * x ** n
    return result

# define a place holder to load x
x = tf.placeholder(tf.float32, [None, 5])
# define a variable 5x1 matrix w with initializing zero
w = tf.Variable(tf.zeros([5, 1]))
# create a tensor that carries matmul with x and w, that is y = XW, [1x5]x[5x1] = [1x1]
y = tf.matmul(x, w)

# define a place holder to load t
t = tf.placeholder(tf.float32, [None, 1])
# define a loss function with sigma((y-t)^2)
loss = tf.reduce_sum(tf.square(y-t))

# create a tensor that minimize the loss function
train_step = tf.train.AdamOptimizer().minimize(loss)

# create session
sess = tf.Session()
# initialize all variables
sess.run(tf.global_variables_initializer())

# create new np array
train_t = np.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4,25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
train_t = train_t.reshape([12, 1])  # convert the array into 12x1 matrix, I think we don't need to reshape the array.

train_x = np.zeros([12, 5])
for row, month in enumerate(range(1, 13)):  # enumerate returns the index(row) and value(month). here 0,1~11,12
    for col, n in enumerate(range(0, 5)):   # it returns the index(col) and value(n). here is 0,0~4,4
        train_x[row][col] = month**n

i = 0
for _ in range(100000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 10000 == 0:
        loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})
        print('Step: %d, Loss: %f' % (i, loss_val))

for _ in range(100000):
    i += 1
    sess.run(train_step, feed_dict={x: train_x, t: train_t})
    if i % 10000 == 0:
        loss_val = sess.run(loss, feed_dict={x: train_x, t: train_t})
        print('Step: %d, Loss: %f' % (i, loss_val))

w_val = sess.run(w)
print(w_val)

def predict(x):
    result = 0.0
    for n in range(0, 5):
        result += w_val[n][0] * x**n
    return result

# predict line and draw figures
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
subplot.set_xlim(1,12)
subplot.scatter(range(1,13), train_t)
linex = np.linspace(1,12,100)   # Return evenly spaced numbers over a specified interval.
liney = predict(linex)
subplot.plot(linex, liney)