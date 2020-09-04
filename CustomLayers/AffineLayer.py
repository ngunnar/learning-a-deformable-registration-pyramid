from __future__ import absolute_import, division, print_function

from .CustomConvLayer import CustomConv
from tensorflow.keras.layers import GlobalAveragePooling3D, Dense, Activation, Concatenate, Layer
from tensorflow.keras.regularizers import Regularizer
import tensorflow.keras.backend as K

class DefReg(Regularizer):
    """Regularizer for the deformation field
    # Arguments
        alpha: Float; regularization factor.
        value: Float; penalize if different than value
    """

    def __init__(self, alpha=1e-5, value=0):
        self.alpha = K.cast_to_floatx(alpha)
        self.value = K.cast_to_floatx(value)

    def __call__(self, x):
        regularization = self.alpha*K.sum(K.abs(x-self.value))
        return regularization

    def get_config(self):
        return {'alpha': float(self.alpha)}


class Affine(Layer):
    def __init__(self, i, affine_regularisation = 1e-1, affine_trainable= True,**kwargs):
        super(Affine, self).__init__(name='affine_{0}'.format(i),**kwargs)
        self.i = i
        self.affine_regularisation = affine_regularisation
        self.affine_trainable = affine_trainable
        self.concat = Concatenate(axis=-1)
        self.globalAveragePooling = GlobalAveragePooling3D()
        self.dense = Dense(12,
                           kernel_initializer='zeros',
                           bias_initializer='zeros',
                           trainable=affine_trainable,
                           activity_regularizer=DefReg(alpha=affine_regularisation,
                                                       value=0))
        self.activation = Activation('linear')
    
    def get_config(self):
        config = super(Affine, self).get_config()
        config.update({"affine_regularisation": self.affine_regularisation, 
                       "affine_trainable": self.affine_trainable,
                       "i": self.i})
        return config
    
    def build(self, input_shape):
        super(Affine, self).build(input_shape)
    
    def call(self, inputs):
        x = self.concat(inputs)
        x = self.globalAveragePooling(x)
        x = self.dense(x)
        x = self.activation(x)
        return x