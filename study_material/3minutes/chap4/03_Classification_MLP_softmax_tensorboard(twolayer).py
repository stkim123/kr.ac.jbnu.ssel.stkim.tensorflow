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



class MLP:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 2], name="X")

        with tf.name_scope('hidden1'):
            # w0 = tf.Variable(tf.random_uniform([2, 3], -1., 1.), name="w0")  # weight
            w0 = tf.Variable(tf.truncated_normal([2, 10]), name="w0")  # weight
            b0 = tf.Variable(tf.zeros([10]), name="b0")  # bias
            hidden1 = tf.nn.tanh(tf.matmul(x, w0) + b0)

        # with tf.name_scope('hidden2'):
        #     # w0 = tf.Variable(tf.random_uniform([2, 3], -1., 1.), name="w0")  # weight
        #     w1 = tf.Variable(tf.truncated_normal([10, 3]), name="w1")  # weight
        #     b1 = tf.Variable(tf.zeros([3]), name="b1")  # bias
        #     hidden2 = tf.nn.tanh(tf.matmul(hidden1, w1) + b1)

        with tf.name_scope('output'):
            # w1 = tf.Variable(tf.random_uniform([3, 1]), name = "w1")
            wo = tf.Variable(tf.zeros([10, 3]), name="wo")
            bo = tf.Variable(tf.zeros([3]), name = "bo")
            p = tf.matmul(hidden1, wo) + bo
            # p = tf.nn.sigmoid(tf.matmul(hidden, wo) + bo, name = "sigmoid")

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 3], name='t')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=p))
                # -tf.reduce_sum(t * tf.log(p) + (1-t)*tf.log(1-p), name='loss')
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            # correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
            correct_prediction = tf.equal(tf.argmax(p), tf.argmax(t))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("weights_hidden", w0)
        tf.summary.histogram("biases_hidden", b0)
        # tf.summary.histogram("weights_output", w1)
        # tf.summary.histogram("biases_output", b1)

        self.x, self.t, self.p = x, t, p
        # self.w0, self.b0, self.w1, self.b1, self.wo, self.bo = w0, b0, w1, b1, wo, bo
        self.w0, self.b0, self.wo, self.bo = w0, b0, wo, bo
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

    def train(self):
        i = 0
        for _ in range(5000):
            i += 1
            self.sess.run(self.train_step, feed_dict={self.x: x_data, self.t: y_data})

            # x_val, w0_val, b0_val, p_val, t_val \
            #     = self.sess.run([self.x, self.w0, self.b0, self.p, self.t], feed_dict={self.x: x_data, self.t: y_data})
            # print("\nx=", x_val, "\nw0=", w0_val, "\nb0=", b0_val, "\np=", p_val, "\nt=", t_val)

            # summary_val, loss_val, acc_val = self.sess.run([self.summary, self.loss, self.accuracy],
            #                                          feed_dict={self.x: x_data, self.t: y_data})
            # print('loss: %f' % loss_val)
            # print('Accuracy:  %f' % acc_val)

            if i % 20 == 0:
                summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy],
                                                         feed_dict={nn.x: x_data, nn.t: y_data})
                print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
                nn.writer.add_summary(summary, i)
        self.loss_val = loss_val
        self.accuracy_val = acc_val
        self.w0_val = self.wo.eval(self.sess)
        self.b0_val = self.b0.eval(self.sess)
        self.wo_val = self.wo.eval(self.sess)
        self.bo_val = self.bo.eval(self.sess)

    def printMLPStructure(self):
        print("input node {}", x_data)
        print("hidden 1 - weight: {}", self.w0_val)
        print("hidden 1 - bias  : {}", self.b0_val)
        print("activation function:" , "relu")
        print("output   - weight: {}", self.wo_val)
        print("output   - bias  : {}", self.bo_val)
        print("use softmax at the last")
        print("use cross entropy as the cost function")
        print("output node {}", y_data)
        print("------------------------------------------")
        print("accuracy:", self.accuracy_val)
        print("loss    :", self.loss_val)

# End of the class definition
#================================================================================
# Execution
nn = MLP()
nn.train()
nn.printMLPStructure()

#================================================================================
# Output
# input node {} [[0 0]
#  [1 0]
#  [1 1]
#  [0 0]
#  [0 0]
#  [0 1]]
# hidden 1 - weight: {} [[-1.15757716  0.62060839  0.91626775]
#  [ 1.00241792 -0.98509157 -0.61357689]
#  [ 0.41410127  0.95940214 -0.74634743]
#  [ 0.97788203  0.89282149 -0.96408993]
#  [-0.66990006 -0.87825483  0.79281199]
#  [-0.85835302  1.13941574  0.54968148]
#  [ 0.78173929  0.76991993 -0.78035843]
#  [-0.58505124 -1.41393697  0.83659142]
#  [-0.83645689  1.09684658  0.5174337 ]
#  [-0.88956231  1.00118816  0.57403183]]
# hidden 1 - bias  : {} [-0.96260875  1.11061072  0.16930139  0.76796389 -0.60053492 -0.97197145
#   0.75297147 -0.25391471 -1.0089184  -0.98945266]
# activation function: relu
# output   - weight: {} [[-1.15757716  0.62060839  0.91626775]
#  [ 1.00241792 -0.98509157 -0.61357689]
#  [ 0.41410127  0.95940214 -0.74634743]
#  [ 0.97788203  0.89282149 -0.96408993]
#  [-0.66990006 -0.87825483  0.79281199]
#  [-0.85835302  1.13941574  0.54968148]
#  [ 0.78173929  0.76991993 -0.78035843]
#  [-0.58505124 -1.41393697  0.83659142]
#  [-0.83645689  1.09684658  0.5174337 ]
#  [-0.88956231  1.00118816  0.57403183]]
# output   - bias  : {} [ 0.44851977 -0.5313623   0.79299641]
# use softmax at the last
# use cross entropy as the cost function
# output node {} [[1 0 0]
#  [0 1 0]
#  [0 0 1]
#  [1 0 0]
#  [1 0 0]
#  [0 0 1]]
# ------------------------------------------
# accuracy: 1.0
# loss    : 0.000618336

