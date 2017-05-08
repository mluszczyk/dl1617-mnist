import json

import numpy
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.saver import Saver

from lab6 import Reshape, Conv, BatchNormalization, Relu, MaxPool, FullyConnected, VariableSaver


class DigitTrainer:
    def train_on_batch(self):
        results = self.sess.run([self.train_step, self.loss, self.accuracy])
        return results[1:]

    def create_model(self, *, trainable):
        initializer = tf.truncated_normal([10, 784], stddev=0.1)
        self.x = tf.Variable(initializer, name='x_learnable')
        self.y_target = tf.stack(tf.one_hot(list(range(10)), 10, dtype=tf.float32))

        layers_list = [
            Reshape([-1, 28, 28, 1]),
            Conv(32),
            BatchNormalization(),
            Relu(),
            MaxPool(),
            Conv(64),
            BatchNormalization(),
            Relu(),
            MaxPool(),
            Reshape([-1, 7 * 7 * 64]),
            FullyConnected(1024),
            Relu(),
            FullyConnected(10)
        ]

        self.variable_saver = VariableSaver()

        signal = self.x
        print('shape', signal.get_shape())
        for idx, layer in enumerate(layers_list):
            signal = layer.contribute(signal, idx, trainable, self.variable_saver.save_variable)
            print('shape', signal.get_shape())

        print(self.variable_saver.var_list)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):

        self.create_model(trainable=False)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        saver = Saver(var_list=self.variable_saver.var_list)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            saver.restore(self.sess, "checkpoint-mnist")

            batches_n = 10000

            losses = []
            try:
                for batch_idx in range(batches_n):
                    vloss = self.train_on_batch()

                    losses.append(vloss)

                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, mean_loss=np.mean(losses[-200:], axis=0))
                        )
                        print('Test results', self.sess.run([self.loss, self.accuracy]))

            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy]))

            data = self.sess.run(self.x)
            with open("digits.json", "w") as f:
                json.dump({"digits": data.tolist()}, f)


if __name__ == '__main__':
    trainer = DigitTrainer()
    trainer.train()
