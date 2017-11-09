# Lab 4 Multi-variable linear regression
import tensorflow as tf
import xlrd

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


tf.set_random_seed(777)  # for reproducibility


file_name = "final_beacondata.xlsx"
workbook = xlrd.open_workbook(file_name)

sheet = workbook.sheet_by_index(3)

rows = sheet.nrows
clos = sheet.ncols


data = [[sheet.cell_value(r, c) for c in range(clos)] for r in range(rows)]
type(data)

txPower_data = []
rssi_data = []
real_distance_data = []

for i in range(1,rows):
    txPower_data.append(sheet.cell_value(i, 0))

for i in range(1,rows):
    rssi_data.append(sheet.cell_value(i, 1))

for i in range(1,rows):
    real_distance_data.append(sheet.cell_value(i, 2))

# placeholders for a tensor that will be always fed.
txPower = tf.placeholder(tf.float32)
rssi = tf.placeholder(tf.float32)
distance = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='txpower_weight1')
Y = tf.Variable(tf.random_normal([1]), name='rssi_weight2')

# hypothesis = 10**((W*txPower - Y*rssi)/10*2)

hypothesis = (W*txPower - Y*rssi)/10*2
print(hypothesis)

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - log10(distance)))

# Minimize. Need a very small learning rate for this data set
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())


for step in range(10001):
    # cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    #                                feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    cost_val, hy_val, _, W_val, Y_val = sess.run([cost, hypothesis, train, W, Y],
                                                 feed_dict={txPower: txPower_data, rssi: rssi_data, distance: real_distance_data})
    if step % 1 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val, "\nW:", W_val, "\nY:", Y_val)