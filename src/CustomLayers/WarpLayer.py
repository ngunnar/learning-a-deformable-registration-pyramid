from __future__ import absolute_import, division, print_function

from packaging import version

#import sys
#sys.path.append('../ext/neuron')
#sys.path.append('../ext/pytools-lib')
#sys.path.append('../ext/pytools-lib/pynd')
#sys.path.append('../ext/pytools-lib/pytools')

import tensorflow as tf
#import Utils.tensorflow_addons as tfimg
import neuron.layers as nrn_layers

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential


class Warp(Layer):
    def __init__(self, name='warp_layer',**kwargs):        
        super(Warp, self).__init__(name=name, **kwargs)
        self.warp = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij',single_transform=False)
    
    def get_config(self):
        config = super(Warp, self).get_config()
        return config
    
    def build(self, input_shape):
        super(Warp, self).build(input_shape)
        self.trainable = False       
    
    def call(self, inputs):
        x = inputs[0]
        flo = inputs[1]
        
        #assert x.shape[0] == flo.shape[0], 'x: {0}, flo: {1}'.format(x.shape, flo.shape)
        assert x.shape[1] == flo.shape[1], 'x: {0}, flo: {1}'.format(x.shape, flo.shape)
        assert x.shape[2] == flo.shape[2], 'x: {0}, flo: {1}'.format(x.shape, flo.shape)
        assert x.shape[3] == flo.shape[3], 'x: {0}, flo: {1}'.format(x.shape, flo.shape)
        assert flo.shape[4] == 3, 'x: {0}, flo: {1}'.format(x.shape, flo.shape)
        
        assert x.shape[1] > 2, x.shape[1]
        assert x.shape[2] > 2, x.shape[2]
        assert x.shape[3] > 2, x.shape[3]
        assert flo.shape[1] > 2, flo.shape[1]
        assert flo.shape[2] > 2, flo.shape[2]
        assert flo.shape[3] > 2, flo.shape[3]
        
        out = self.warp([x, flo])
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input    

    warp = Warp()
    config = warp.get_config()
    print(config)

    img = Input(batch_shape=(16, 256, 256, 256, 1))
    flow = Input(batch_shape=(16, 256, 256 , 256, 3))
    out = warp([img, flow])    
    model = Model([img, flow], out)
    print(model.summary()) 