
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import scale

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

with open('./20171016_BeaconDataSet.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, delimiter=',', unpack=True, dtype='float32')


# txPower,# RSSI,# R.DISTANCE
# data = np.loadtxt('./featurecsv4.csv', delimiter=',',
#                   unpack=True, dtype='float32')

# x_data = Txpower, rssi
# y_data = distance
txPower_data = np.transpose(data[0:1])
txPower_data = scale(txPower_data)  # data standization
rssi_data = np.transpose(data[1:2])
rssi_data = scale(rssi_data)  # data standization
distance_data = np.transpose(data[2:])

class MultiLinearRegression:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            #x = tf.placeholder(tf.float32, [None, 2], name="X")
            txPower = tf.placeholder(tf.float32, [None, 1], name="X")
            rssi = tf.placeholder(tf.float32, [None, 1], name="X")

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([1]), name='txpower_weight1')
            Y = tf.Variable(tf.truncated_normal([1]), name='rssi_weight2')
            predicted4Learning = (W * txPower - Y * rssi) / 10 * 2
            computedDistance = tf.pow(10., (txPower - rssi)/20.)

            # wo = tf.Variable(tf.truncated_normal([15, 1]), name="wo")
            # bo = tf.Variable(tf.zeros([1]), name = "bo")
            # p = tf.matmul(x, wo) + bo

        with tf.name_scope('optimizer'):
            distance = tf.placeholder(tf.float32, [None, 1], name='t')

            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(t - p)))
            loss = tf.reduce_sum(tf.square(log10(distance) - predicted4Learning))
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):

            # correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
            # correct_prediction = tf.equal(tf.argmax(p), tf.argmax(t))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            accuracy = loss

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("weights_hidden", W)
        tf.summary.histogram("biases_hidden", Y)

        # tf.summary.histogram("weights_output", w1)
        # tf.summary.histogram("biases_output", b1)

        self.txPower, self.rssi, self.predicted, self.distance = txPower, rssi, predicted4Learning, distance
        self.computedDistance = computedDistance
        self.W, self.Y = W, Y
        self.train_step = train_step
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
            self.sess.run(self.train_step,\
                          feed_dict={self.txPower: txPower_data,\
                                     self.rssi: rssi_data,\
                                     self.distance: distance_data})

            # summary_val, loss_val, acc_val, p_val = \
            #     self.sess.run([self.summary, self.loss, self.accuracy, self.predicted], \
            #                   feed_dict={self.txPower: txPower_data, \
            #                              self.rssi: rssi_data, \
            #                              self.distance: distance_data})
            #
            # print("\nx=", x_val, "\np=", p_val, "\nt=", t_val)

            summary_val, loss_val, acc_val, p_val, computedD_val = \
                self.sess.run([self.summary, self.loss, self.accuracy, self.predicted, self.computedDistance], \
                              feed_dict={self.txPower: txPower_data, \
                                         self.rssi: rssi_data, \
                                         self.distance: distance_data})

            # print('loss: %f' % loss_val)
            # print('Accuracy:  %f' % acc_val)

            if i % 50 == 0:
                summary_val, loss_val, acc_val, p_val, computedD_val = \
                    self.sess.run([self.summary, self.loss, self.accuracy, self.predicted, self.computedDistance], \
                                  feed_dict={self.txPower: txPower_data, \
                                             self.rssi: rssi_data, \
                                             self.distance: distance_data})
                print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
                # print('\np=', p_val)
                nn.writer.add_summary(summary_val, i)

        self.loss_val = loss_val
        self.accuracy_val = acc_val
        self.W_val = self.W.eval(self.sess)
        self.Y_val = self.Y.eval(self.sess)
        self.p_val = p_val
        self.computedDistance_val = computedD_val

    def printMLPStructure(self):
        # print("input node {}", txPower_data)
        # print("input node {}", rssi_data)
        print("Weight - W: {}", self.W_val)
        print("Weight - Y: {}", self.Y_val)
        print("distance node {}", self.computedDistance_val)
        # print("output node {}", distance_data)
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
