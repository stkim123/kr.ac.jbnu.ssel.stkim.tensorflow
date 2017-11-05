# minimum likelyhood estimation

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame, Series


# prepare training data set
np.random.seed(20160512)
n0, mu0, variance0 = 20, [10, 11], 20
                                                            # multivariate_normal: Draw random samples from a multivariate normal distribution.
data0 = multivariate_normal(mu0, np.eye(2)*variance0, n0)   # np.eye :Return a 2-D array with ones on the diagonal and zeros elsewhere.
df0 = DataFrame(data0, columns=['x1', 'x2'])
df0['t'] = 0                                                # set all 't' column to be 0

n1, mu1, variance1 = 15, [18, 20], 22
data1 = multivariate_normal(mu1, np.eye(2)*variance1 ,n1)
df1 = DataFrame(data1, columns=['x1','x2'])
df1['t'] = 1

df = pd.concat([df0, df1], ignore_index=True)               # concatnate all training set
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)    # shuffle the index like a real


train_x = train_set[['x1', 'x2']].as_matrix()
train_t = train_set['t'].as_matrix().reshape([len(train_set), 1])


# define a place holder to load x
x = tf.placeholder(tf.float32, [None, 2])
# define a variable 2x1 matrix w with initializing zero
w = tf.Variable(tf.zeros([2, 1]))       # weight
w0 = tf.Variable(tf.zeros([1]))    # bias

# estimation
f = tf.matmul(x, w) + w0
p = tf.sigmoid(f)


# define a place holder to load t
t = tf.placeholder(tf.float32, [None, 1])

# define a loss function with sigma((y-t)^2)
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))

# create a tensor that minimize the loss function
train_step = tf.train.AdamOptimizer().minimize(loss)

# define accuracy
correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create session
sess = tf.Session()
# initialize all variables
sess.run(tf.global_variables_initializer())

i = 0
for _ in range(20000):
    i += 1
    sess.run(train_step, feed_dict={x: train_x, t: train_t})
    if i % 2000 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: train_x, t: train_t})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

w0_val, w_val = sess.run([w0, w])
w0_val, w1_val, w2_val = w0_val[0], w_val[0][0], w_val[1][0]
print(w0_val, w1_val, w2_val)
#
# ##########################################################################
train_set0 = train_set[train_set['t']==0]
train_set1 = train_set[train_set['t']==1]

fig = plt.figure(figsize=(6, 6))
subplot = fig.add_subplot(1, 1, 1)
subplot.set_ylim([0, 30])
subplot.set_xlim([0, 30])
subplot.scatter(train_set1.x1, train_set1.x2, marker='x')
subplot.scatter(train_set0.x1, train_set0.x2, marker='o')

linex = np.linspace(0, 30, 10)
liney = - (w1_val*linex/w2_val + w0_val/w2_val)

subplot.plot(linex, liney)

field = [[(1 / (1 + np.exp(-(w0_val + w1_val*x1 + w2_val*x2))))
          for x1 in np.linspace(0,30,100)]
         for x2 in np.linspace(0,30,100)]

subplot.imshow(field, origin='lower', extent=(0,30,0,30),
               cmap=plt.cm.gray_r, alpha=0.5)