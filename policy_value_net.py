# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from params import *
from chess_utils import raise_error
import os
import sys


PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

class policy_value_net(object):

    def __init__(self, learning_rate, num_of_res_block=19):
        self.save_path = "./gpu_models"
        self.logging = True
        
        if not os.path.exists(self.save_path):
            raise_error(__file__, sys._getframe().f_lineno, \
                message=("save path -> %s does not exists"%(self.save_path)))
            os.makedirs(self.save_path)
        
        # multi gpus
        self.strategy = tf.distribute.MirroredStrategy()
        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        self.distributed_train = lambda it: self.strategy.experimental_run(self.train_step, it)
        self.distributed_train = tf.function(self.distributed_train)

        with tf.device('/gpu:0'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with self.strategy.scope():

            # Variables
            self.filters_size = 128 # 256
            self.prob_size = 2086
            self.digest = None

            self.inputs_ = tf.keras.layers.Input([9, 10, 14], dtype='float32', name='inputs')
            self.l2 = 0.0001
            self.momentum = 0.99

            self.layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, strides=(1, 1), padding='same')(self.inputs_)
            self.layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=self.momentum)(self.layer)
            self.layer = tf.keras.layers.ReLU()(self.layer)

            # residual_block
            with tf.name_scope("residual_block"):
                for _ in range(num_of_res_block):
                    self.layer = self.residual_block(self.layer)

            # policy_head
            with tf.name_scope("policy_head"):
                self.policy_head = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=(1, 1), padding='same')(self.layer)
                self.policy_head = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=self.momentum)(self.policy_head)
                self.policy_head = tf.keras.layers.ReLU()(self.policy_head)

                self.policy_head = tf.keras.layers.Reshape([9 * 10 * 2])(self.policy_head)
                self.policy_head = tf.keras.layers.Dense(self.prob_size)(self.policy_head)

            # value_head
            with tf.name_scope("value_head"):
                self.value_head = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same')(self.layer)
                self.value_head = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=self.momentum)(
                    self.value_head)
                self.value_head = tf.keras.layers.ReLU()(self.value_head)

                self.value_head = tf.keras.layers.Reshape([9 * 10 * 1])(self.value_head)
                self.value_head = tf.keras.layers.Dense(256, activation='relu')(self.value_head)
                self.value_head = tf.keras.layers.Dense(1, activation='tanh')(self.value_head)

            self.model = tf.keras.Model(
                inputs=[self.inputs_],
                outputs=[self.policy_head, self.value_head])

            self.model.summary()

            # optimizer & loss
            self.momentum_opt = 0.9
            self.learning_rate = learning_rate
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum_opt, nesterov=True)

            self.compute_accuracy = tf.keras.metrics.CategoricalAccuracy()
            self.avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

            self.checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
            self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

            # Restore variables on creation if a checkpoint exists.
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def residual_block(self, in_layer):
        orig = tf.convert_to_tensor(in_layer)
        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, strides=(1, 1), padding='same')(in_layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=self.momentum)(layer)
        layer = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=self.filters_size, strides=(1, 1), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=self.momentum)(layer)
        add_layer = tf.keras.layers.add([orig, layer])
        out = tf.keras.layers.ReLU()(add_layer)

        return out

    def save(self):
        with self.strategy.scope():
            self.checkpoint.save(self.checkpoint_prefix)
        
    def compute_metrics(self, pi_, policy_head):
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(input=policy_head, axis=1), tf.argmax(input=pi_, axis=1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(input_tensor=correct_prediction, name='accuracy')
        return accuracy

    def apply_regularization(self, regularizer, weights_list=None):
        if not weights_list:
            raise_error(__file__, sys._getframe().f_lineno, message=("No weights to regularize"))
            raise ValueError('No weights to regularize.')
        # with tf.name_scope('get_regularization_penalty', values=weights_list) as scope:
        with tf.name_scope('get_regularization_penalty') as scope:
            penalties = [regularizer(w) for w in weights_list]
            penalties = [p if p is not None else tf.constant(0.0) for p in penalties]
            for p in penalties:
                if p.get_shape().ndims != 0:
                    error_msg = 'regularizer must return a scalar Tensor instead of a Tensor with rank %d.'%p.get_shape().ndims
                    raise_error(__file__, sys._getframe().f_lineno, message=error_msg)
                    raise ValueError(error_msg)

            summed_penalty = tf.add_n(penalties, name=scope)
            return summed_penalty

    def compute_loss(self, pi_, z_, policy_head, value_head):

        with tf.name_scope("loss"):
            policy_loss = tf.keras.losses.categorical_crossentropy(y_true=pi_, y_pred=policy_head, from_logits=True)
            policy_loss = tf.reduce_mean(policy_loss)

            value_loss = tf.keras.losses.mean_squared_error(z_, value_head)
            value_loss = tf.reduce_mean(value_loss)

            regularizer = tf.keras.regularizers.l2(self.l2)
            regular_variables = self.model.trainable_variables
            l2_loss = self.apply_regularization(regularizer, regular_variables)

            self.loss = value_loss + policy_loss + l2_loss

        return self.loss

    def train_step(self, it, learning_rate=0):
        positions = it[0]
        pi = it[1]
        z = it[2]

        if True:
            with tf.GradientTape() as tape:
                policy_head, value_head = self.model(positions, training=True)
                loss = self.compute_loss(pi, z, policy_head, value_head)
                self.compute_accuracy(pi, policy_head)
                self.avg_loss(loss)
            grads = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return self.compute_accuracy.result(), self.avg_loss.result(), self.global_step

    def forward(self, positions):

        with self.strategy.scope():
            positions=np.array(positions)
            if len(positions.shape) == 3:
                sp = positions.shape
                positions=np.reshape(positions, [1, sp[0], sp[1], sp[2]])
            action_probs, value = self.model(positions, training=False)

        return action_probs, value


if __name__ == '__main__':
    if test_mode:
        net = policy_value_net(learning_rate=0.0001, num_of_res_block=9)
