import tensorflow as tf


def weight_variable(shape, *, trainable, stddev=0.1):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initializer, name='weight', trainable=trainable)


def bias_variable(shape, *, trainable, bias=0.1):
    initializer = tf.constant(bias, shape=shape)
    return tf.Variable(initializer, name='bias', trainable=trainable)


class Relu:
    def contribute(self, signal, idx, trainable, save_variable):
        return tf.nn.relu(signal)


class Reshape:
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def contribute(self, signal, idx, trainable, save_variable):
        return tf.reshape(signal, self.output_shape)


class Conv:
    def __init__(self, output_channels: int):
        self.output_channels = output_channels

    def contribute(self, signal, idx, trainable, save_variable):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        assert len(signal.get_shape()) == 4
        input_channels = int(signal.get_shape()[3])

        with tf.variable_scope('conv_' + str(idx + 1)):
            W_conv1 = weight_variable([5, 5, input_channels, self.output_channels], trainable=trainable)
            save_variable(W_conv1)
            b_conv1 = bias_variable([self.output_channels], trainable=trainable)
            save_variable(b_conv1)

        return conv2d(signal, W_conv1) + b_conv1


class MaxPool:
    def contribute(self, signal, idx, trainable, save_variable):
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
        return max_pool_2x2(signal)


class BatchNormalization:
    def contribute(self, signal, idx, trainable, save_variable):
        input_shape = signal.get_shape()

        with tf.variable_scope('batch_norm_' + str(idx + 1)):
            gamma = weight_variable([int(input_shape[-1])], trainable=trainable)
            save_variable(gamma)
            beta = bias_variable([int(input_shape[-1])], trainable=trainable)
            save_variable(beta)

        assert len(input_shape) == 4
        mean = tf.reduce_mean(signal, axis=[0, 1, 2])
        assert len(mean.get_shape()) == 1
        stdvarsq = tf.reduce_mean((signal - mean) ** 2, axis=[0, 1, 2])
        assert len(stdvarsq.get_shape()) == 1
        eps = 1e-5
        normalized = ((signal - mean) / tf.sqrt(stdvarsq + eps))
        assert (str(normalized.get_shape()) == str(input_shape))
        return tf.multiply(gamma, normalized) + beta


class FullyConnected:
    def __init__(self, neuron_num):
        self.neuron_num = neuron_num

    def contribute(self, signal, idx, trainable, save_variable):
        cur_num_neurons = int(signal.get_shape()[1])
        stddev = 0.1
        with tf.variable_scope('fc_' + str(idx + 1)):
            W_fc = weight_variable([cur_num_neurons, self.neuron_num], stddev=stddev, trainable=trainable)
            save_variable(W_fc)
            b_fc = bias_variable([self.neuron_num], bias=0.1, trainable=trainable)
            save_variable(b_fc)

        signal = tf.matmul(signal, W_fc) + b_fc
        return signal
