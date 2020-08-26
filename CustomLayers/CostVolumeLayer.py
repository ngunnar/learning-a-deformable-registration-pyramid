from __future__ import absolute_import, division, print_function

from tensorflow.keras.layers import LeakyReLU, Concatenate
from tensorflow.keras import layers
import tensorflow as tf

class CostVolume(layers.Layer):
    def __init__(self, search_range,**kwargs):
        super(CostVolume, self).__init__(**kwargs)
        self.search_range = search_range
        self.concat = Concatenate(axis=4)
        self.leaky_relu = LeakyReLU(alpha = 0.1)
        
    def build(self, input_shape):
        super(CostVolume, self).build(input_shape)
        self.trainable = False
        
    def call(self, inputs):
        c1 = inputs[0]
        warp = inputs[1]
        padded_lvl = tf.pad(warp, [[0, 0], 
                                   [self.search_range, self.search_range], 
                                   [self.search_range, self.search_range], 
                                   [self.search_range, self.search_range], 
                                   [0, 0]])
        _, d, h, w, _ = tf.unstack(tf.shape(c1))
        max_offset = self.search_range * 2 + 1
        
        cost_vol = []
        for z in range(0, max_offset):
            for y in range(0, max_offset):
                for x in range(0, max_offset):
                    slice = tf.slice(padded_lvl, [0, z, y, x, 0], [-1, d, h, w, -1])
                    cost = tf.reduce_mean(c1 * slice, axis=4, keepdims=True)
                    cost_vol.append(cost)
        cost_vol = self.concat(cost_vol)
        cost_vol = self.leaky_relu(cost_vol)
        return cost_vol
