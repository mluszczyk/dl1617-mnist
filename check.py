"""This script prints model's output for the matrices generated by digits.py

This is for sanity check.
"""


import json

import tensorflow as tf
from tensorflow.python.training.saver import Saver

from model import inner_model, CHECKPOINT_FILE_NAME


class DigitChecker:
    def create_model(self, *, trainable):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')

        self.y_prob, self.var_list = inner_model(trainable, self.x)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def eval(self):

        with open("digits.json") as f:
            digits_data = json.load(f)['digits']

        self.create_model(trainable=False)

        saver = Saver(var_list=self.var_list)

        with tf.Session() as self.sess:
            saver.restore(self.sess, CHECKPOINT_FILE_NAME)

            print('Test results', self.sess.run(self.y_prob, feed_dict={self.x: digits_data}))


if __name__ == '__main__':
    trainer = DigitChecker()
    trainer.eval()

