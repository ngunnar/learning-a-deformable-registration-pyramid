from __future__ import absolute_import, division, print_function

from .CustomConvLayer import CustomConv
from tensorflow.keras.layers import Layer

class Pyramid(Layer):
    def __init__(self, gamma, num,**kwargs):
        super(Pyramid, self).__init__(name='pyramid',**kwargs)
        #f = [16, 32, 64, 96, 128, 196]
        #f = [4, 8, 16, 32, 64, 96]
        self.filters = [16,32,32,32,32]
        assert num <= len(self.filters)
        self.num = num
        self.conv_a = []
        self.conv_aa = []
        self.conv_b = []
        for i in range(num):
            self.conv_a.append(CustomConv(filters=self.filters[i],strides=2, gamma=gamma, name='conv{0}a'.format(i+1)))
            self.conv_aa.append(CustomConv(filters=self.filters[i],strides=1, gamma=gamma, name='conv{0}aa'.format(i+1)))
            self.conv_b.append(CustomConv(filters=self.filters[i],strides=1, gamma=gamma, name='conv{0}b'.format(i+1)))
    
    def get_config(self):
        config = super(Pyramid, self).get_config()
        config.update({"filters": self.filters,
               "num": self.num})
        return config
    
    def build(self, input_shape):
        super(Pyramid, self).build(input_shape)
        
    def call(self, inputs):
        c = inputs
        out = [c]
        for i in range(self.num):
            c = self.conv_b[i](self.conv_aa[i](self.conv_a[i](c)))
            out.append(c)
        return out