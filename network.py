import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (
    InputLayer,
    DenseLayer,
    DeConv2d,
    ReshapeLayer,
    BatchNormLayer,
    Conv2d,
    FlattenLayer
)

class GAN(object):
    def __init__(self, class_num, images_size, channel=3, code_dim=128):
        # fake picture class
        self.class_num = class_num
        # input && output image size n x n
        self.images_size = images_size
        # input && output channel
        self.channel = channel
        # code dimension for noise to input 
        self.code_dim = code_dim
        # placeholder for noise input
        self.z = tf.placeholder(tf.float32, [None, code_dim], name='z_noise')
        self.label_class = tf.placeholder(tf.int32, [None, ], name='label_class')
        self.real_images =  tf.placeholder(tf.float32, [None, images_size, images_size, channel], name='real_images')
        self.network = self.generator(self.z, self.label_class)

    def generator(self, z, label_class, is_train=True, reuse=False):
        # NOTE: concate z & label might be wrong, need to test
        labels_one_hot = tf.one_hot(label_class, self.class_num)
        z_labels = tf.concat([z, labels_one_hot], 1)
        image_size = self.images_size
        s16 = image_size // 16
        gf_dim = 64    # Dimension of gen filters in first conv layer. [64]
        c_dim = self.channel    # n_color 3
        w_init = tf.glorot_normal_initializer()
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("generator", reuse=reuse):
            net_in = InputLayer(z_labels, name='g/in')
            net_h0 = DenseLayer(net_in, n_units=(gf_dim * 8 * s16 * s16), W_init=w_init,
                    act = tf.identity, name='g/h0/lin')
            net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
            net_h0 = BatchNormLayer(net_h0, decay=0.9, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h0/batch_norm')

            net_h1 = DeConv2d(net_h0, gf_dim * 4, (5, 5), strides=(2, 2),
                    padding='SAME', act=None, W_init=w_init, name='g/h1/decon2d')
            net_h1 = BatchNormLayer(net_h1, decay=0.9, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h1/batch_norm')

            net_h2 = DeConv2d(net_h1, gf_dim * 2, (5, 5), strides=(2, 2),
                    padding='SAME', act=None, W_init=w_init, name='g/h2/decon2d')
            net_h2 = BatchNormLayer(net_h2, decay=0.9, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h2/batch_norm')

            net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), strides=(2, 2),
                    padding='SAME', act=None, W_init=w_init, name='g/h3/decon2d')
            net_h3 = BatchNormLayer(net_h3, decay=0.9, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h3/batch_norm')

            net_h4 = DeConv2d(net_h3, c_dim, (5, 5), strides=(2, 2),
                    padding='SAME', act=None, W_init=w_init, name='g/h4/decon2d')
            net_h4.outputs = tf.nn.tanh(net_h4.outputs)
        return net_h4

    def discriminator(self, inputs, is_train=True, reuse=False):
        df_dim = image_size = self.images_size   # Dimension of discrim filters in first conv layer. [64]
        w_init = tf.glorot_normal_initializer()
        gamma_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)
        with tf.variable_scope("discriminator", reuse=reuse):
            
            net_in = InputLayer(inputs, name='d/in')
            net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lrelu,
                    padding='SAME', W_init=w_init, name='d/h0/conv2d')

            net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='d/h1/conv2d')
            net_h1 = BatchNormLayer(net_h1, decay=0.9, act=lrelu,
                    is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

            net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='d/h2/conv2d')
            net_h2 = BatchNormLayer(net_h2, decay=0.9, act=lrelu,
                    is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

            net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='d/h3/conv2d')
            net_h3 = BatchNormLayer(net_h3, decay=0.9, act=lrelu,
                    is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

            net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
            net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                    W_init = w_init, name='d/h4/lin_sigmoid')
            logits = net_h4.outputs
            net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
        return net_h4, logits