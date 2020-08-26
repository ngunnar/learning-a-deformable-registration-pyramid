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
        f = [64, 64, 32, 16, 8]
        self.first = first
        self.concatenate = Concatenate(axis = 4)
        self.dec = []
        for j in range(len(f)):
            self.dec.append(CustomConv(filters=f[j],strides=1, gamma=gamma, name='decoder{0}_{1}'.format(i,j)))
        
        #self.dec1 = CustomConv(filters=f[0],strides=1, gamma=gamma, name='decoder{0}_1'.format(i))
        #self.dec2 = CustomConv(filters=f[1],strides=1, gamma=gamma, name='decoder{0}_2'.format(i))
        #self.dec3 = CustomConv(filters=f[2],strides=1, gamma=gamma, name='decoder{0}_3'.format(i))
        #self.dec4 = CustomConv(filters=f[3],strides=1, gamma=gamma, name='decoder{0}_4'.format(i))
        #self.dec5 = CustomConv(filters=f[4],strides=1, gamma=gamma, name='decoder{0}_5'.format(i))
        self.predict_flow = Conv3D(filters = 3,
                                   kernel_size = (3,3,3),
                                   strides=1,
                                   kernel_initializer=he_normal(),
                                   kernel_regularizer=l2(gamma),
                                   padding='same')
        self.denseNet = denseNet
    
    def build(self, input_shape):
        super(OpticalFlowEstimator, self).build(input_shape)
    
    def call(self, inputs):
        x = inputs[0]
        c1 = inputs[1]
        up_flow_prev = inputs[2]
        up_feat_prev = inputs[3]
        #print(x.shape, c1.shape)
        #if up_flow_prev is not None:
        #    print(up_flow_prev.shape, up_feat_prev.shape)
        # Optical Flow Estimator
        if self.first == False:
            x = self.concatenate([x, c1, up_flow_prev, up_feat_prev])            
        
        if self.denseNet:
            # DenseNet
            for j in range(len(self.dec)):
                x = self.concatenate([self.dec[j](x), x])
            #x = self.concatenate([self.dec2(x), x])
            #x = self.concatenate([self.dec3(x), x])
            #x = self.concatenate([self.dec4(x), x])
            #upfeat = self.concatenate([self.dec5(x), x])
        else:
            for j in range(len(self.dec)):
                x = self.dec[j](x)
            #x = self.dec1(x)
            #x = self.dec2(x)
            #x = self.dec3(x)
            #x = self.dec4(x)
            #upfeat = self.dec5(x)        
                
        flow = self.predict_flow(x)
        return [x, flow]