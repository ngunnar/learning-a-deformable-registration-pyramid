from __future__ import absolute_import, division, print_function

from .CustomConvLayer import CustomConv
from tensorflow.keras.layers import Conv3D, Add, Layer
from tensorflow.keras.regularizers import l2

class Context(Layer):
    def __init__(self, gamma, i,**kwargs):
        super(Context, self).__init__(name='context_{0}'.format(i),**kwargs)
        #f = [128, 128, 128, 96, 64, 32, 3]
        self.filters = [64, 64, 64, 32, 16, 8, 3]
        self.i = i
        
        self.conv1 = CustomConv(filters = self.filters[0], strides=1, gamma=gamma, name='dc_conv1_{0}'.format(self.i), dilation = 1)
        self.conv2 = CustomConv(filters = self.filters[1], strides=1, gamma=gamma, name='dc_conv2_{0}'.format(self.i), dilation = 2)        
        self.conv3 = CustomConv(filters = self.filters[2], strides=1, gamma=gamma, name='dc_conv3_{0}'.format(self.i), dilation = 4)        
        self.conv4 = CustomConv(filters = self.filters[3], strides=1, gamma=gamma, name='dc_conv4_{0}'.format(self.i), dilation = 8)        
        self.conv5 = CustomConv(filters = self.filters[4], strides=1, gamma=gamma, name='dc_conv5_{0}'.format(self.i), dilation = 16)        
        self.conv6 = CustomConv(filters = self.filters[5], strides=1, gamma=gamma, name='dc_conv6_{0}'.format(self.i))
        self.conv7 = Conv3D(filters = self.filters[6],
                            kernel_size=(3,3,3),
                            strides=1, 
                            padding='same',
                            use_bias=True,
                            kernel_regularizer=l2(gamma),
                            bias_regularizer=l2(gamma),
                            name='dc_conv7_{0}'.format(i))
    
        self.add8 = Add(name ='add_{0}'.format(self.i))
    
    def get_config(self):
        config = super(Context, self).get_config()
        config.update({"filters": self.filters,
               "i": self.i})
        return config
    
    def build(self, input_shape):
        super(Context, self).build(input_shape)
    
    def call(self, inputs):
        upfeat = inputs[0]
        flow = inputs[1]
        x = self.conv1(upfeat)        
        x = self.conv2(x)        
        x = self.conv3(x)        
        x = self.conv4(x)        
        x = self.conv5(x)        
        x = self.conv6(x)        
        x = self.conv7(x)      
        
        x = self.add8([flow,x])
        return x