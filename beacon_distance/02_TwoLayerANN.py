
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import scale


with open('./20171016_BeaconDataSet.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, delimiter=',', unpack=True, dtype='float32')


# data = np.loadtxt('./featurecsv4.csv', delimiter=',',
#                   unpack=True, dtype='float32')

# x_data = 0, 1
# y_data = 2, 3, 4
x_data = np.transpose(data[0:2])
# x_data = scale(x_data)  # data standization
y_data = np.transpose(data[2:])

class MultiLinearRegression:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 2], name="X")

        with tf.name_scope('hidden1'):
            w0 = tf.Variable(tf.truncated_normal([2, 10]), name="w0")  # weight
            b0 = tf.Variable(tf.zeros([10]), name="b0")  # bias
            hidden1 = tf.nn.tanh(tf.matmul(x, w0) + b0)

        with tf.name_scope('hidden2'):
            w1 = tf.Variable(tf.truncated_normal([10, 20]), name="w0")  # weight
            b1 = tf.Variable(tf.zeros([20]), name="b0")  # bias
            hidden2 = tf.nn.tanh(tf.matmul(hidden1, w1) + b1)

        with tf.name_scope('hidden3'):
            w2 = tf.Variable(tf.truncated_normal([20, 10]), name="w0")  # weight
            b2 = tf.Variable(tf.zeros([10]), name="b0")  # bias
            hidden3 = tf.nn.tanh(tf.matmul(hidden2, w2) + b2)

        with tf.name_scope('output'):
            wo = tf.Variable(tf.truncated_normal([10, 1]), name="wo")
            bo = tf.Variable(tf.zeros([1]), name = "bo")
            p = tf.matmul(hidden3, wo) + bo

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 1], name='t')

            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(t - p)))
            loss = tf.reduce_sum(tf.square(t - p))
            # train_step = tf.train.AdamOptimizer().minimize(loss)
            train_step = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)

        with tf.name_scope('evaluator'):
            allowed_range_upper_bound = t + t * tf.constant([[0.02]])
            allowed_range_lower_bound = t - t * tf.constant([[0.02]])

            corr_down = tf.greater_equal(tf.sqrt(tf.square(p - t)), allowed_range_lower_bound, name="greator")
            corr_up = tf.less_equal(tf.sqrt(tf.square(p - t)), allowed_range_upper_bound, name="greator")
            correct_prediction = tf.logical_and(corr_up, corr_down)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

            # correct_prediction = tf.equal(tf.sign(p-t), tf.sign(t-0.5))
            # correct_prediction = tf.equal(tf.argmax(p), tf.argmax(t))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            # accuracy = loss

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("weights_hidden", wo)
        tf.summary.histogram("biases_hidden", bo)
        # tf.summary.histogram("weights_output", w1)
        # tf.summary.histogram("biases_output", b1)

        self.x, self.t, self.p = x, t, p
        self.allowed_range = allowed_range_upper_bound
        # self.w0, self.b0, self.w1, self.b1, self.wo, self.bo = w0, b0, w1, b1, wo, bo
        # self.w0, self.b0, self.wo, self.bo = w0, b0, wo, bo
        self.wo, self.bo = wo, bo
        self.train_step = train_step
        self.correct_prediction = correct_prediction
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()

        logpath = os.path.join("D:\\", "Temp2", "beacon_distance_prediction")
        writer = tf.summary.FileWriter(logpath, sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer

    def train(self):
        i = 0
        for _ in range(10000):
            i += 1
            self.sess.run(self.train_step, feed_dict={self.x: x_data, self.t: y_data})

            x_val, wo_val, bo_val, p_val, t_val, correct_predict_val, allowed_range_val \
                = self.sess.run([self.x, self.wo, self.bo, self.p, self.t, self.correct_prediction, self.allowed_range], feed_dict={self.x: x_data, self.t: y_data})
            # print("\nx=", x_val, "\np=", p_val, "\nt=", t_val, "\ncorrect_prediction=", correct_predict_val, "\nallowed_range=", allowed_range_val)

            summary_val, loss_val, acc_val, p_val = self.sess.run([self.summary, self.loss, self.accuracy, self.p],
                                                     feed_dict={self.x: x_data, self.t: y_data})
            # print('loss: %f' % loss_val)
            # print('Accuracy:  %f' % acc_val)

            if i % 50 == 0:
                summary, loss_val, acc_val, p_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy, nn.p],
                                                         feed_dict={nn.x: x_data, nn.t: y_data})
                print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
                # print('\np=', p_val)
                nn.writer.add_summary(summary, i)

        self.loss_val = loss_val
        self.accuracy_val = acc_val
        self.wo_val = self.wo.eval(self.sess)
        self.bo_val = self.bo.eval(self.sess)
        self.p_val = p_val
        self.correct_predict_val = correct_predict_val
        self.allowed_range = allowed_range_val

    def printMLPStructure(self):
        print("input node {}", x_data)
        print("output   - weight: {}", self.wo_val)
        print("output   - bias  : {}", self.bo_val)
        print("use softmax at the last")
        print("use cross entropy as the cost function")
        print("prediction node {}", self.p_val)
        print("output node {}", y_data)
        print("------------------------------------------")
        print("accuracy:", self.accuracy_val)
        print("loss    :", self.loss_val)

# End of the class definition
#================================================================================
# Execution
nn = MultiLinearRegression()
nn.train()
nn.printMLPStructure()

#================================================================================
# Output
