from __future__ import absolute_import, division, print_function

from tensorflow.keras.layers import Conv3D, LeakyReLU, Layer, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal

class CustomConv(Layer):
    def __init__(self,filters, strides, name, gamma, dilation = 1,**kwargs):
        super(CustomConv, self).__init__(name=name,**kwargs)
        self.filters = filters
        self.strides = strides
        self.gamma = gamma
        self.dilation = dilation
        self.conv3d = Conv3D(filters = self.filters, 
                             kernel_size = (3,3,3), 
                             strides=self.strides, 
                             dilation_rate = self.dilation, 
                             padding='same', 
                             kernel_initializer=he_normal(),
                             kernel_regularizer=l2(gamma))
        
        self.leakyRelu = LeakyReLU(alpha = 0.2)
        self.batchNormalization = BatchNormalization() 

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({"filters": self.filters,
               "strides": self.strides,
               "gamma": self.gamma,
               "dilation": self.dilation})
        return config
    
    def build(self, input_shape):
        super(CustomConv, self).build(input_shape)    
    
    def call(self, inputs):        
        x = self.conv3d(inputs)
        x = self.leakyRelu(x)
        x = self.batchNormalization(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return self.conv2d.compute_output_shape(input_shape)