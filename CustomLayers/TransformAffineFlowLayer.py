from tensorflow.keras.layers import Layer, Lambda
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class TransformAffineFlow(Layer):
    def __init__(self, shape):
        super(TransformAffineFlow, self).__init__()
        self.shape = shape
        
    def get_config(self):
        config = super(TransformAffineFlow, self).get_config()
        config.update({"shape": self.shape})
    def add_identity(x):
        identity = np.array([[1,0,0,0,0,1,0,0,0,0,1,0]], 'float32')
        identity = tf.convert_to_tensor(identity)
        x = x + identity
        return x
    
    #def init_flow(shape):
    #    return tf.zeros(shape,dtype='float32')  
    
    def get_ones(shape):
        return tf.ones(shape,dtype='float32')  
        
    def call(self, inputs):
        a_flow = inputs[0]
        batch_size = tf.shape(a_flow)[0]
        flow = inputs[1]
        #if len(inputs) > 1:
        #    flow = inputs[1]
        #else:
        #    flow = Lambda(TransformAffineFlow.init_flow)((batch_size, *self.shape))
        ones = Lambda(TransformAffineFlow.get_ones)((batch_size, *self.shape[0:3], 1))
        a_flow = Lambda(TransformAffineFlow.add_identity)(a_flow)
        a_flow = tf.reshape(a_flow, (batch_size, 3, 4))
        flow = tf.keras.layers.Concatenate(axis=-1)([flow, ones])
        flow = tf.reshape(flow, (batch_size, np.prod(flow.shape[1:4]), 4))
        flow = tf.matmul(flow, a_flow, transpose_b=True)
        flow = tf.reshape(flow, (batch_size, *self.shape))
        return flow