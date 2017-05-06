import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data
import os
 
''''

Link to lecture slides: 
https://docs.google.com/presentation/d/1Vh8NPCWkgVy_I79aqjHyDnlp7TNmVfaEqfIc4LAM-JM/edit?usp=sharing


Tasks:
1. Check that the given implementation reaches 95% test accuracy for
   architecture input-64-64-10 in a few thousand batches.

2. Improve initialization and check that the network learns much faster
   and reaches over 97% test accuracy.

3. Check, that with proper initialization we can train architecture input-64-64-64-64-64-10,
   while with bad initialization it does not even get off the ground.

4. If you do not feel comfortable enough with training networks and/or tensorflow I suggest adding
dropout implemented in tensorflow (check documentation, new placeholder will be needed to indicate train/test phase).

5. Check that with 10 hidden layers (64 units each) even with proper initialization
   the network has a hard time to start learning.

6. Implement batch normalization (use train mode also for testing - it should perform well enough):
    * compute batch mean and variance as tensorflow operations,
    * add new variables beta and gamma to scale and shift the result,
    * check that the networks learns much faster for 5 layers (even though training time per batch is a bit longer),
    * check that the network learns even for 10 hidden layers.

Bonus task:

Design and implement in tensorflow (by using tensorflow functions) a simple convnet and achieve 99% test accuracy.

Note:
This is an exemplary exercise. MNIST dataset is very simple and we are using it here to get resuts quickly.
To get more meaningful experience with training convnets use the CIFAR dataset.
'''


def weight_variable(shape, stddev=0.1):
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initializer, name='weight')


def bias_variable(shape, bias=0.1):
    initializer = tf.constant(bias, shape=shape)
    return tf.Variable(initializer, name='bias')


class FullyConnected:
    def __init__(self, neuron_num):
        self.neuron_num = neuron_num

    def contribute(self, signal, idx):
        cur_num_neurons = int(signal.get_shape()[1])
        stddev = 0.1
        with tf.variable_scope('fc_' + str(idx + 1)):
            W_fc = weight_variable([cur_num_neurons, self.neuron_num], stddev)
            b_fc = bias_variable([self.neuron_num], 0.1)

        signal = tf.matmul(signal, W_fc) + b_fc
        return signal


class Relu:
    def contribute(self, signal, idx):
        return tf.nn.relu(signal)


class Reshape:
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def contribute(self, signal, idx):
        return tf.reshape(signal, self.output_shape)


class Conv:
    def __init__(self, output_channels: int):
        self.output_channels = output_channels

    def contribute(self, signal, idx):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        assert len(signal.get_shape()) == 4
        input_channels = int(signal.get_shape()[3])

        with tf.variable_scope('conv_' + str(idx + 1)):
            W_conv1 = weight_variable([5, 5, input_channels, self.output_channels])
            b_conv1 = bias_variable([self.output_channels])

        return conv2d(signal, W_conv1) + b_conv1


class MaxPool:
    def contribute(self, signal, idx):
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
        return max_pool_2x2(signal)


class BatchNormalization:
    def contribute(self, signal, idx):
        input_shape = signal.get_shape()

        with tf.variable_scope('batch_norm_' + str(idx + 1)):
            gamma = weight_variable([int(input_shape[-1])])
            beta = bias_variable([int(input_shape[-1])])

        assert len(input_shape) == 4
        mean = tf.reduce_mean(signal, axis=[0, 1, 2])
        assert len(mean.get_shape()) == 1
        stdvarsq = tf.reduce_mean((signal - mean) ** 2)
        eps = 1e-5
        normalized = ((signal - mean) / tf.sqrt(stdvarsq + eps))
        return tf.multiply(gamma, normalized) + beta



class MnistTrainer:
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})
        return results[1:]

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 10])

        layers_list = [
            Reshape([-1, 28, 28, 1]),
            BatchNormalization(),
            Conv(16),
            Relu(),
            MaxPool(),
            BatchNormalization(),
            Conv(32),
            Relu(),
            MaxPool(),
            BatchNormalization(),
            Conv(64),
            Relu(),
            MaxPool(),
            BatchNormalization(),
            Conv(128),
            Relu(),
            MaxPool(),
            Reshape([-1, 2 * 2 * 128]),
            FullyConnected(64),
            Relu(),
            FullyConnected(10)
        ]

        signal = self.x
        print('shape', signal.get_shape())
        for idx, layer in enumerate(layers_list):
            signal = layer.contribute(signal, idx)
            print('shape', signal.get_shape())

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):

        self.create_model()
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)
 
                    vloss = self.train_on_batch(batch_xs, batch_ys)
 
                    losses.append(vloss)
 
                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, mean_loss=np.mean(losses[-200:], axis=0))
                        )
                        print('Test results', self.sess.run([self.loss, self.accuracy],
                                                            feed_dict={self.x: mnist.test.images,
                                                                       self.y_target: mnist.test.labels}))

 
            except KeyboardInterrupt:
                print('Stopping training!')
                pass
 
            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                self.y_target: mnist.test.labels}))
 
 
if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()


