
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import scale



with open('./featurecsv4.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, delimiter=',', unpack=True, dtype='float32')

# data = np.loadtxt('./featurecsv4.csv', delimiter=',',
#                   unpack=True, dtype='float32')

# x_data = 0, 1
# y_data = 2, 3, 4
x_data = np.transpose(data[0:15])
x_data = scale(x_data)  # data standization
y_data = np.transpose(data[15:])


class MultiLinearRegression:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 15], name="X")

        with tf.name_scope('output'):
            wo = tf.Variable(tf.truncated_normal([15, 1]), name="wo")
            bo = tf.Variable(tf.zeros([1]), name = "bo")
            p = tf.matmul(x, wo) + bo

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 1], name='t')
            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(p - t)))

            total_error = tf.reduce_sum(tf.square(t - tf.reduce_mean(t)))
            unexplained_error = tf.reduce_sum(tf.square(t- p))
            R_squared = tf.subtract(1., tf.div(total_error, unexplained_error))

            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(t - p)))
            loss = tf.reduce_sum(tf.square(t - p))
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            # correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
            # correct_prediction = tf.equal(tf.argmax(p), tf.argmax(t))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            accuracy = loss

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("weights_hidden", wo)
        tf.summary.histogram("biases_hidden", bo)
        # tf.summary.histogram("weights_output", w1)
        # tf.summary.histogram("biases_output", b1)

        self.x, self.t, self.p = x, t, p
        # self.w0, self.b0, self.w1, self.b1, self.wo, self.bo = w0, b0, w1, b1, wo, bo
        # self.w0, self.b0, self.wo, self.bo = w0, b0, wo, bo
        self.wo, self.bo = wo, bo
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        self.R_squared = R_squared

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()

        logpath = os.path.join("D:\\", "Temp2", "mrl_instant_feedback")
        writer = tf.summary.FileWriter(logpath, sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer

    def train(self):
        i = 0
        for _ in range(50000):
            i += 1
            self.sess.run(self.train_step, feed_dict={self.x: x_data, self.t: y_data})

            # x_val, wo_val, bo_val, p_val, t_val \
            #     = self.sess.run([self.x, self.wo, self.bo, self.p, self.t], feed_dict={self.x: x_data, self.t: y_data})
            # print("\nx=", x_val, "\np=", p_val, "\nt=", t_val)
            #
            summary_val, loss_val, R_squared_val, acc_val, p_val = self.sess.run([self.summary, self.loss, self.R_squared, self.accuracy, self.p],
                                                     feed_dict={self.x: x_data, self.t: y_data})
            # print('loss: %f' % loss_val)
            # print('Accuracy:  %f' % acc_val)

            if i % 20 == 0:
                summary, loss_val, acc_val, p_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy, nn.p],
                                                         feed_dict={nn.x: x_data, nn.t: y_data})
                print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
                # print('\np=', p_val)
                nn.writer.add_summary(summary, i)

        self.loss_val = loss_val
        self.accuracy_val = acc_val
        self.wo_val = self.wo.eval(self.sess)
        self.bo_val = self.bo.eval(self.sess)
        self.R_squared_val = R_squared_val
        self.p_val = p_val

    def printMLPStructure(self):
        # print("input node {}", x_data)
        print("output   - weight: {}", self.wo_val)
        print("output   - bias  : {}", self.bo_val)
        print("use softmax at the last")
        print("use cross entropy as the cost function")
        print("prediction node {}", self.p_val)
        # print("output node {}", y_data)
        print("------------------------------------------")
        print("accuracy:", self.accuracy_val)
        print("loss    :", self.loss_val)
        print("R_squared_val:", self.R_squared_val)

# End of the class definition
#================================================================================
# Execution
nn = MultiLinearRegression()
nn.train()
nn.printMLPStructure()

#================================================================================
# Output
