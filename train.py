import tensorflow as tf
import numpy as np
from network import GAN
from data import data_provider
import time

class Train(object):
    def __init__(self, mode='train', epochs=10, log_step=1):
        self.epochs = epochs
        self.log_step = log_step
        self._data = data_provider()
        self.step_per_epoch = self._data.step_per_epoch
        self.steps = epochs * self.step_per_epoch
        self._net = self._build_network()
        # TODO: add GPU mode && necessary config in session
        self.sess = tf.Session()
        self._data = data_provider()
        if mode == 'train':
            self.sess.run(tf.global_variables_initializer())
            
        
    def train(self):
        start_time = time.time()
        for step in range(self.steps):
            noise_z, real_images, label_real = self._data.next_batch
            # optimize D
            _, batch_d_loss, d_summary = self.sess.run([self._net.d_optim, self._net.d_loss,
                                                        self._net.d_merged_summary],
                                                        feed_dict={
                                                            self._net.z:noise_z,
                                                            self._net.label_class:label_real,
                                                            self._net.real_images:real_images
                                                        })
            # optimize G twice
            _, batch_g_loss = self.sess.run([self._net.d_optim, self._net.g_loss],
                                                    feed_dict={
                                                        self._net.z:noise_z,
                                                        self._net.label_class:label_real,
                                                        self._net.real_images:real_images
                                                        })
            _, batch_g_loss, category_loss, g_cheat_loss, \
                                        g_summary = self.sess.run([self._net.d_optim, self._net.g_loss, self._net.category_loss, self._net.g_cheat_loss, self._net.g_merged_summary],
                                                    feed_dict={
                                                        self._net.z:noise_z,
                                                        self._net.label_class:label_real,
                                                        self._net.real_images:real_images
                                                        })
            if step % self.log_step == 0:
                log = "Epoch:[ %d / %d ], step: %d, d_loss: %.4f, g_loss: %.4f , time_using: %d " % (step / self.epochs, self.epochs, step, batch_d_loss, batch_g_loss, time.time() - start_time)
                print(log)
            # TODO: checkpoint, save model summary restore model
    

    def _build_network(self):
        return GAN(class_num=10, images_size=256)