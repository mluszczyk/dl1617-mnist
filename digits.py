"""Generate file digits.json containing the images that maximize the probability
for each digit.
"""


import json

import numpy as np
import tensorflow as tf
from tensorflow.python.training.saver import Saver

from model import create_model, CHECKPOINT_FILE_NAME


class DigitTrainer:
    def train_on_batch(self):
        results = self.sess.run([self.train_step, self.loss, self.accuracy])
        return results[1:]

    def create_model(self, *, trainable):
        initializer = tf.truncated_normal([10, 784], stddev=0.1)
        self.x = tf.Variable(initializer, name='x_learnable')
        self.y_target = tf.stack(tf.one_hot(list(range(10)), 10, dtype=tf.float32))

        self.var_list, self.loss, self.accuracy, self.train_step, y_prob = create_model(trainable, self.x, self.y_target)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):

        self.create_model(trainable=False)

        saver = Saver(var_list=self.var_list)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            saver.restore(self.sess, CHECKPOINT_FILE_NAME)

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
