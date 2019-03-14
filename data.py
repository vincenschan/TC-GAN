import numpy as np

class data_provider(object):
    def __init__(self, batch_size=16, code_dim=128, images_size=256, channel=3, class_num=10):
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.images_size = images_size
        self.channel = channel
        self.class_num = class_num
    
    # data get format: noise_z, real_images, label_real
    @property
    def next_batch(self):
        return  np.ones([self.batch_size, self.code_dim]), \
                np.ones([self.batch_size, self.images_size, self.images_size, self.channel]), \
                np.ones([self.batch_size, ])
    
    @property
    def step_per_epoch(self):
        return 10000