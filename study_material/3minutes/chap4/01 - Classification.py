# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 합니다.
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# define a place holder to load x
# x = tf.placeholder(tf.float32, [None, 2])
# y = tf.placeholder(tf.float32, [None, 2])
X = tf.placeholder(tf.float32, name ="X")
Y = tf.placeholder(tf.float32, name ="Y")

# define a variable 2x1 matrix w with initializing zero
W0 = tf.Variable(tf.random_uniform([2, 3], -1., 1.), name="W0")       # weight
b0 = tf.Variable(tf.zeros([3]), name="b0")    # bias

# estimation
f = tf.matmul(X, W0) + b0

# f_act = tf.tanh(f)
# W1 = tf.Variable(tf.random_uniform([3, 1]))
# b1 = tf.Variable(tf.zeros([1]))
# p = tf.nn.softmax(tf.matmul(f_act, W1) + b1)
# cost = -tf.reduce_sum(Y * tf.log(p))

p = tf.nn.softmax(f)
cost = -tf.reduce_sum(Y * tf.log(p))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

i = 0
for _ in range(100000):
    i += 1
    sess.run(train_step, feed_dict={X:x_data, Y:y_data})
    if i % 100 == 0:
        cost_val = sess.run([cost], feed_dict={X:x_data, Y:y_data})
        print(i, cost_val)

