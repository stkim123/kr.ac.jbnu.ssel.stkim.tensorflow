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
# 신경망은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정합니다.
# W0 = tf.Variable(tf.random_uniform([2, 3], -1., 1.), name="W0")       # weight
# b0 = tf.Variable(tf.zeros([3]), name="b0")    # bias

W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

# f = tf.matmul(X, W0) + b0
# hidden = tf.nn.tanh(f)

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)
model = tf.nn.softmax(L)

# ????????????? why it does not have next layer?
# W1 = tf.Variable(tf.random_uniform([3, 1]))
# b1 = tf.Variable(tf.zeros([1]))
# p = tf.nn.softmax(tf.matmul(hidden, W1) + b1)

# cost = -tf.reduce_sum(Y*tf.log(p) + (1-Y)*tf.log(1-p))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.
# 예) [[0 1 0] [1 0 0]] -> [1 0]
#    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
