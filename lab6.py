"""Train the model on MNIST dataset."""

import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.saver import Saver

from model import create_model, CHECKPOINT_FILE_NAME


class MnistTrainer:
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})
        return results[1:]

    def create_model(self, *, trainable):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 10])

        self.var_list, self.loss, self.accuracy, self.train_step, y_prob = create_model(trainable, self.x, self.y_target)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):

        self.create_model(trainable=True)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        saver = Saver(var_list=self.var_list)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables

            if os.path.exists(CHECKPOINT_FILE_NAME + ".meta"):
                print("Restoring existing weights")
                saver.restore(self.sess, CHECKPOINT_FILE_NAME)
            else:
                print("Training a new model")

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

                        saver.save(self.sess, CHECKPOINT_FILE_NAME)
 
            except KeyboardInterrupt:
                print('Stopping training!')
                pass
 
            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                self.y_target: mnist.test.labels}))


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()


