from __future__ import absolute_import, division, print_function

from .CustomConvLayer import CustomConv
from tensorflow.keras.layers import Layer, Conv3D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
import tensorflow as tf

class OpticalFlowEstimator(Layer):    
    def __init__(self, i, gamma, denseNet, first = False,**kwargs):
        super(OpticalFlowEstimator, self).__init__(name='opt_flow_est_{0}'.format(i),**kwargs)
        #f = [128, 128, 96, 64, 32, 3]
        #f = [64, 64, 32, 16, 8, 3]
        self.i = i
        self.filters = [64, 64, 32, 16, 8]
        self.first = first
        self.concatenate = Concatenate(axis = 4)
        self.dec = []
        for j in range(len(self.filters)):
            self.dec.append(CustomConv(filters=self.filters[j],strides=1, gamma=gamma, name='decoder{0}_{1}'.format(self.i,j)))
        
        self.predict_flow = Conv3D(filters = 3,
                                   kernel_size = (3,3,3),
                                   strides=1,
                                   kernel_initializer=he_normal(),
                                   kernel_regularizer=l2(gamma),
                                   padding='same')
        self.denseNet = denseNet
    
    def get_config(self):
        config = super(OpticalFlowEstimator, self).get_config()
        config.update({"filters": self.filters,
               "first": self.first,
               "denseNet": self.denseNet,
               "i": self.i})
        return config    
    
    def build(self, input_shape):
        super(OpticalFlowEstimator, self).build(input_shape)
    
    def call(self, inputs):
        x = inputs[0]
        c1 = inputs[1]
        init_flow = inputs[2]
        
        # Optical Flow Estimator
        if self.first == False and len(inputs) > 3:
            up_feat_prev = inputs[3]
            x = self.concatenate([x, c1, init_flow, up_feat_prev])          
        else:
            x = self.concatenate([x, c1, init_flow])
        if self.denseNet:
            # DenseNet
            for j in range(len(self.dec)):
                x = self.concatenate([self.dec[j](x), x])
        else:
            for j in range(len(self.dec)):
                x = self.dec[j](x)
                
        flow = self.predict_flow(x)
        return [x, flow]