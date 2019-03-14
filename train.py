import tensorflow as tf
import numpy as np
from network import GAN
from data import data_provider


class Train(object):
    def __init__(self, mode='train'):
        self._data = data_provider()
        self._net = self._build_network()
        # TODO: add GPU mode && necessary config in session
        self.sess = tf.Session()
        if mode == 'train':
            self.sess.run(tf.global_variables_initializer())
        
    def train(self):
        pass
    
    def _build_network(self):
        return GAN(class_num=10, images_size=256)