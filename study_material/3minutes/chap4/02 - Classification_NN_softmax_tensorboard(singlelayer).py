# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np
import os

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

class SingleLayerNetwork:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 2], name="X")

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.truncated_normal([2, 3]), name="w0")  # weight
            b0 = tf.Variable(tf.zeros([3]), name="b0")  # bias
            output = tf.nn.relu(tf.matmul(x, w0) + b0)
            p = tf.nn.softmax(output)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 3], name='t')
            loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(p), axis=1), name='loss')
            train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
            # train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            # correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
            correct_prediction = tf.equal(tf.argmax(p), tf.argmax(t))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("weights_hidden", w0)
        tf.summary.histogram("biases_hidden", b0)

        self.x, self.t, self.p = x, t, p
        self.w0, self.b0  = w0, b0,
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()

        logpath = os.path.join("D:\\", "Temp2", "mnist_sl_logs")
        writer = tf.summary.FileWriter(logpath, sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer


nn = SingleLayerNetwork()

i = 0
for _ in range(2000):
    i += 1
    nn.sess.run(nn.train_step, feed_dict={nn.x: x_data, nn.t: y_data})

    x_val, w0_val, b0_val, p_val, t_val \
        = nn.sess.run([nn.x, nn.w0, nn.b0, nn.p, nn.t], feed_dict={nn.x: x_data, nn.t: y_data})
    print(x_val, "\n", w0_val, "\n", b0_val, "\n", p_val, "\n", t_val)
    print("x={}\n w0={}\n b0={}\n p={}\n t={}".format(x_val, w0_val, b0_val, p_val, t_val))
        # x_val, "\n", w0_val, "\n", b0_val, "\n", p_val, "\n", t_val)

    summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy],
                                             feed_dict={nn.x: x_data, nn.t: y_data})
    print('loss: %f' % loss_val)
    print('Accuracy:  %f' % acc_val)

    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy],
            feed_dict={nn.x:x_data, nn.t: y_data})
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)

