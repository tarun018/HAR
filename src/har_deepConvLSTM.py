import tensorflow as tf

import lasagne
import theano
import time

import numpy as np
import cPickle as cp #serializing and de-serializing a Python object structure
import theano.tensor as T
from sliding_window import sliding_window

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

#Load the OPPORTUNITY processed dataset. Sensor data is segmented using a sliding window of fixed length. The class associated with each segment corresponds to the gesture which has been observed during that interval. Given a sliding window of length T, we choose the class of the sequence as the label at t=T, or in other words, the label of last sample in the window.
# In[14]:

def load_dataset(filename):

    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('gestures.data')

assert NB_SENSOR_CHANNELS == X_train.shape[1]
def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

# Sensor data is segmented using a sliding window mechanism
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
# Data is reshaped since the input of the network is a 4 dimension tensor
X_test = X_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, 1))

# Sensor data is segmented using a sliding window mechanism
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
# Data is reshaped since the input of the network is a 4 dimension tensor
X_train = X_train.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, 1))


def one_hot(label):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    label_num = len(label)
    new_label = label.reshape(label_num)  # shape : [sample_num]
    # because max is 5, and we will create 6 columns
    n_values = np.max(new_label) + 1
    return np.eye(n_values)[np.array(new_label, dtype=np.int32)]
y_test=one_hot(y_test)
y_train=one_hot(y_train)

print("data is ready")

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, rnnW, rnnB, lstm_size):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='VALID'))
    # l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
    #                     strides=[1, 2, 2, 1], padding='SAME')
    # l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='VALID'))
    # l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
    #                     strides=[1, 2, 2, 1], padding='SAME')
    # l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2a, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='VALID'))
    # l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
    #                     strides=[1, 2, 2, 1], padding='SAME')
    # l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    # l3 = tf.nn.dropout(l3, p_keep_conv)
    l4a = tf.nn.relu(tf.nn.conv2d(l3a, w4,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='VALID'))
    shuff = tf.transpose(l4a, [1, 0, 2, 3])
    shp1 = tf.reshape(shuff, [-1, lstm_size])
    X_split = tf.split(shp1, 8, 0) # split them to time_step_size (28 arrays)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    # Stack two LSTM layers, both layers has the same shape
    lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)

    outputs, _states = tf.contrib.rnn.static_rnn(lstm_layers, X_split, dtype=tf.float32)

    print("tf net end")

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], rnnW) + rnnB

X = tf.placeholder("float", [None, 24, 113, 1])
Y = tf.placeholder("float", [None, 18])

lstm_size = 128
w = init_weights([5, 1, 1, 64])       # 3x3x1 conv, 32 outputs
w2 = init_weights([5, 1, 64, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([5, 1, 64, 64])    # 3x3x32 conv, 128 outputs
w4 = init_weights([5, 1, 64, 64]) # FC 128 * 4 * 4 inputs, 625 outputs
rnnW = init_weights([lstm_size, 128])
rnnB = init_weights([128])
pre_Y = model(X, w, w2, w3, w4, rnnW, rnnB,lstm_size)
print ("get cnn output");

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre_Y, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(pre_Y, 1)

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
print("net work done")

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(X_train), BATCH_SIZE), range(BATCH_SIZE, len(X_train)+1, BATCH_SIZE)):
            sess.run(train_op, feed_dict={X: X_train[start:end], Y: y_train[start:end]})

        test_indices = np.arange(len(X_test))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: 100]

        print(i, np.mean(np.argmax(y_test[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: X_test[test_indices]})))
