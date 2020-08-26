from __future__ import absolute_import, division, print_function

import tensorflow as tf
import Utils.tensorflow_addons as tfimg
from tensorflow.keras.layers import Layer
import neuron.layers as nrn_layers

class Resize(Layer):
    def __init__(self, scalar, factor, name, **kwargs):
        super(Resize, self).__init__(name=name, **kwargs)
        self.scalar = scalar
        self.resize = nrn_layers.Resize(zoom_factor=factor)
        
    def build(self, input_shape):
        super(Resize, self).build(input_shape)
        
    def call(self, inputs):
        return self.resize(inputs) * self.scalar