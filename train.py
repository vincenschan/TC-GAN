import tensorflow as tf
import numpy as np
from network import GAN
from data import data_provider


class Train(object):
    def __init__(self):
        self._data = data_provider()
        self._net = self._build_network()

    def train(self):
        pass
    
    def _build_network(self):
        return GAN()