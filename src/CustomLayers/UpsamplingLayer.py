from __future__ import absolute_import, division, print_function

from tensorflow.keras.layers import Layer, UpSampling3D, Conv3D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal

class Upsampling(Layer):
    def __init__(self, gamma, i,**kwargs):
        super(Upsampling, self).__init__(name='upsampling_{0}'.format(i),**kwargs)  
        self.gamma = gamma
        self.i = i
        self.upsampling_flow = UpSampling3D(size = 2)
        self.conv3d_flow = Conv3D(filters = 3,
                                  kernel_size= 3,                                   
                                  strides = 1,
                                  padding='same',
                                  kernel_initializer=he_normal(),
                                  kernel_regularizer=l2(self.gamma))
        self.upsampling_feat = UpSampling3D(size = 2)
        self.conv3d_feat = Conv3D(filters = 32,
                                  kernel_size= 3,                                  
                                  strides = 1,
                                  padding='same',
                                  kernel_initializer=he_normal(),
                                  kernel_regularizer=l2(self.gamma))
    def get_config(self):
        config = super(Upsampling, self).get_config()
        config.update({"gamma": self.gamma,
               "i": self.i})
        return config    
    def build(self, input_shape):
        super(Upsampling, self).build(input_shape)
        
    def call(self, inputs):
        flow = inputs[0]
        up_flow = self.upsampling_flow(flow)
        up_flow = self.conv3d_flow(up_flow)
        
        if len(inputs) > 1:
            feat = inputs[1]
            up_feat = self.upsampling_feat(feat)
            up_feat = self.conv3d_feat(up_feat)
            return up_flow, up_feat
        return up_flow
        