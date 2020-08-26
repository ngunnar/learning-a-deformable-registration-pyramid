import sys
sys.path.append('./ext/neuron')
sys.path.append('./ext/pytools-lib')
sys.path.append('./ext/pytools-lib/pynd')
sys.path.append('./ext/pytools-lib/pytools')

import tensorflow as tf
import numpy as np
from CustomLayers.PyramidLayer import Pyramid
from CustomLayers.WarpLayer import Warp
from CustomLayers.OpticalFlowEstimatorLayer import OpticalFlowEstimator
from CustomLayers.CostVolumeLayer import CostVolume
from CustomLayers.UpsamplingLayer import Upsampling
from CustomLayers.ContextLayer import Context
from CustomLayers.ResizeLayer import Resize
from tensorflow.keras.layers import Concatenate

from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from neuron.layers import VecInt, Negate

import tensorflow as tf
import numpy as np

class PWC_model(Model):
    def __init__(self, inputs, outputs, name, config):        
        self.shape = (int(config['depth']), int(config['height']), int(config['width']))        
        super(PWC_model, self).__init__(inputs=inputs, outputs=outputs, name=name)
        
def create_model(config, name):
    useDenseNet = config['use_dense_net']
    useContextNet = config['use_context_net']
    batch_size = config['batch_size']
    depth = config['depth']
    height = config['height']
    width = config['width']
    gamma = config['gamma']
    lowest = config['lowest']
    last = config['last']
    cost_search_range = config['cost_search_range']
    use_atlas = config['use_atlas']
    
    ## Initiazing Layers
    pyramid = Pyramid(gamma=gamma, num = lowest)
    cost = CostVolume(search_range=cost_search_range)
    
    warps = []
    flow_ests = []
    ups = []
    contexs = []
    for i in range(lowest + 1):
        if i == lowest:
            first = True
        else:
            warps.append(Warp(name='warp_{0}'.format(i+1)))
            first = False
        flow_ests.append(OpticalFlowEstimator(i=i, gamma=gamma, denseNet= useDenseNet, first = first))
        if i != last:
            ups.append(Upsampling(i=i, gamma= gamma))
        
        if useContextNet or i == last:
            contexs.append(Context(i=i, gamma=gamma))

    scalar = 2**(last)
    resize = Resize(scalar = scalar, factor=scalar, name='est_flow')

    ### Creating model ##
    fixed = tf.keras.layers.Input(shape=(depth, height, width, 1), name='fixed_img')
    moving = tf.keras.layers.Input(shape=(depth, height, width, 1), name='moving_img')
    
    inputs = [fixed, moving]
    if use_atlas:
        moving_seg = tf.keras.layers.Input(shape=(depth, height, width, 1), name='moving_seg')
        inputs.append(moving_seg)
        
    out1 = pyramid(fixed)
    out2 = pyramid(moving)
    
    up_flow, up_feat = None, None
    outputs = []    
    #for i in range(1, lowest - last + 2):
    for i in range(1, lowest - last + 3):
        l = lowest + 1 - i
             
        if i != 1:
            s = 1.0#2.0
            warp = warps[l-1]([out2[l], up_flow * s])
        else:
            warp = out2[l]
        
        cv = cost([out1[l], warp])
        [upfeat, flow] = flow_ests[l]([cv, out1[l], up_flow, up_feat])
        if useContextNet:
            flow = contexs[l]([upfeat, flow])
        elif i == last:
            flow = contexs[0]([upfeat, flow])        
        flow = Lambda(lambda x:x, name = "est_flow{0}".format(l))(flow)
        
        if l != 0:
            outputs.append(flow)
            r = Resize(scalar = 1.0, factor = 1/(2**l), name='p_score_{0}'.format(l))
            warp_moving = Warp(name='warp_m_{0}'.format(i+1))([r(moving), flow])
            #o = Concatenate(axis=-1, name='sim_{0}'.format(l))([out1[l], warp])
            o = Concatenate(axis=-1, name='sim_{0}'.format(l))([r(fixed), warp_moving])
            outputs.append(o)
        if l != 0:
            up_flow, up_feat = ups[l-1]([flow, upfeat])
    
    flow_est = flow
    #flow_est = resize(flow)
    flow_int = VecInt(method='ss', name='flow_int', int_steps=7)(flow_est)
    #flow_neg = Negate()(flow_est)
    #flow_neg = VecInt(method='ss', name='neg_flow-int', int_steps=7)(flow_neg)
    
    warped = Warp(name='sim')([moving, flow_est])    
    outputs.append(flow_est)
    outputs.append(warped)
        
    if use_atlas:
        warped_moving_seg = Warp(name='seg')([moving_seg, flow_int])
        outputs.append(warped_moving_seg)
    
    return PWC_model(inputs=inputs, outputs=outputs, name=name, config = config)


if __name__ == "__main__":
    config = {'use_dense_net':True, 'use_context_net':True, 'batch_size':10, 'depth':256, 'height':256, 'width':256, 'gamma': 1e-5, 'lowest': 4, 'last':1}
    pwc_model = create_model(config = config, name="PWC_Net")
    print(pwc_model.summary())
