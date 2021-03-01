import sys
sys.path.append('./src/ext/neuron')
sys.path.append('./src/ext/pytools-lib')
sys.path.append('./src/ext/pytools-lib/pynd')
sys.path.append('./src/ext/pytools-lib/pytools')

import tensorflow as tf
import numpy as np
from .CustomLayers.PyramidLayer import Pyramid
from .CustomLayers.WarpLayer import Warp
from .CustomLayers.OpticalFlowEstimatorLayer import OpticalFlowEstimator
from .CustomLayers.CostVolumeLayer import CostVolume
from .CustomLayers.UpsamplingLayer import Upsampling
from .CustomLayers.ContextLayer import Context
from .CustomLayers.ResizeLayer import Resize
from .CustomLayers.AffineLayer import Affine
from .CustomLayers.TransformAffineFlowLayer import TransformAffineFlow
from tensorflow.keras.layers import Concatenate
from .losses import NCC, Affine_loss, Grad, Dice

from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

import tensorflow as tf
import numpy as np

from neuron.layers import VecInt

def create_model(config, name):
    
    def w_loss(loss):
        def l(_, yp):
            i = yp[..., :yp.shape[-1]//2]
            w = yp[..., yp.shape[-1]//2:]
            return loss(i, w)
        return l
    

    lowest = config['lowest']
    last = config['last']
    pyramid_filters = config['pyramid_filters']
    
    useDenseNet = config['use_dense_net']
    useContextNet = config['use_context_net']
    depth = config['depth']
    height = config['height']
    width = config['width']
    d = config['d']
    use_atlas = config['use_atlas']
    label_classes = config['label_classes']
    use_affine = config['use_affine']
    use_def = config['use_def']
    deform_filters = config['deform_filters']
    
    d_l = config['sim_loss']
    assert d_l in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    
    if d_l in ['ncc', 'cc']:
        d_l = NCC().loss
    else:
        #d_l = tf.keras.losses.MeanSquaredError(reduction='none')
        d_l = tf.keras.losses.MeanSquaredError()
    
    ## Initiazing Layers
    pyramid = Pyramid(gamma=0.001, num = lowest, filters=pyramid_filters)
    cost = CostVolume(search_range=d)
    
    warps_1 = []
    warps_2 = []
    flow_ests = []
    affines = []
    ups = []
    contexs = []
    for i in range(lowest + 1):
        if i == lowest:
            first = True
            warps_2.append(Warp(name='warp2_{0}'.format(i+1)))
        else:
            warps_1.append(Warp(name='warp1_{0}'.format(i+1)))
            warps_2.append(Warp(name='warp2_{0}'.format(i+1)))
            first = False
        if use_def:
            flow_ests.append(OpticalFlowEstimator(i=i, gamma=0.001, denseNet= useDenseNet, filters=deform_filters, first = first))
        if use_affine:
            affines.append(Affine(i=i))
        if i != last:
            ups.append(Upsampling(i=i, gamma= 0.001))
        
        if useContextNet or i == last:
            contexs.append(Context(i=i, gamma= 0.001))

    ### Creating model ##
    fixed = tf.keras.layers.Input(shape=(depth, height, width, 1), name='fixed_img')
    moving = tf.keras.layers.Input(shape=(depth, height, width, 1), name='moving_img')
    batch_size = tf.shape(fixed, name='batch_size')[0]
    
    inputs = [fixed, moving]
    if use_atlas:
        moving_seg = tf.keras.layers.Input(shape=(depth, height, width, label_classes), name='moving_seg')
        inputs.append(moving_seg)
    
    out1 = pyramid(fixed)
    out2 = pyramid(moving)
    
    up_flow, up_feat = None, None
    flow = None
    outputs = []
    loss = []
    loss_weights = []
    for i in range(1, lowest - last + 3):
        l = lowest + 1 - i
        s = 2.0
        x_dim = depth // (2**l)
        y_dim = height // (2**l)
        z_dim = width // (2**l)
        shape = (x_dim, y_dim, z_dim, 3)
        if i != 1:
            flow = up_flow * s
            warp = warps_1[l-1]([out2[l], flow])
        else:
            flow = Lambda(lambda x: tf.zeros(x, dtype='float32'))((batch_size, *shape))
            warp = out2[l]
        
        if use_affine:
            A = affines[l]([out1[l], warp])
            A = Lambda(lambda x:x, name = "A_flow{0}".format(l))(A)
            outputs.append(A)
            loss.append(Affine_loss().loss)
            loss_weights.append(config['alphas'][l])
            
            flow = TransformAffineFlow(shape)([A, flow])
            #outputs.append(flow)
            #loss.append(Grad(config['smooth_loss']).loss)
            #loss_weights.append(config['alphas'][l])
            warp = warps_2[l]([out2[l], flow]) #or warp, flow
            
        if use_def:
            cv = cost([out1[l], warp])
        
            if i != 1 and use_def:
                [upfeat, flow] = flow_ests[l]([cv, out1[l], flow, up_feat])
            else:
                [upfeat, flow] = flow_ests[l]([cv, out1[l], flow])       
                
            if useContextNet:
                flow = contexs[l]([upfeat, flow])
            elif i == last:
                flow = contexs[0]([upfeat, flow])
            
            flow = Lambda(lambda x:x, name = "def_flow{0}".format(l))(flow)
            
            outputs.append(flow)
            loss.append(Grad(config['smooth_loss']).loss)
            loss_weights.append(config['betas'][l])
        
        #if use_affine and use_def:
        #    flow_final = TransformAffineFlow(shape)([A, flow_def])
        #elif use_def:
        #    flow_final = flow_def
        #else:
        #    zero_flow = Lambda(lambda x: tf.zeros(x, dtype='float32'))((batch_size, *shape))
        #    flow_final = TransformAffineFlow(shape)([A, zero_flow])
        
        #flow = Lambda(lambda x:x, name = "final_flow{0}".format(l))(flow)
        if l != 0:
            r = Resize(scalar = 1.0, factor = 1/(2**l), name='p_score_{0}'.format(l))
            warp_moving = Warp(name='warp_m_{0}'.format(i+1))([r(moving), flow])
            o = Concatenate(axis=-1, name='sim_{0}'.format(l))([r(fixed), warp_moving])
            outputs.append(o)
            loss.append(w_loss(d_l))
            loss_weights.append(config['gamma'][l])
        if l != 0:
            if use_def:
                up_flow, up_feat = ups[l-1]([flow, upfeat])
            else:
                up_flow = ups[l-1]([flow])
    
    warped = Warp(name='sim')([moving, flow])    
    outputs.append(warped)
    loss.append(d_l)
    loss_weights.append(config['gamma'][0])
    
    if use_atlas:
        #flow_int = VecInt(method='ss', name='flow_int', int_steps=7)(flow)
        #moving_seg = tf.one_hot(moving_seg, label_classes)
        #if moving_seg.dtype != tf.int32:
        #    moving_seg = tf.cast(moving_seg, tf.int32)
        warped_moving_seg =  Warp(name='seg')([moving_seg, flow])
        outputs.append(warped_moving_seg)
        loss.append(Dice(label_classes).loss)
        loss_weights.append(config['lambda'])
    
    return Model(inputs=inputs, outputs=outputs, name=name), loss, loss_weights

if __name__ == "__main__":
    config = {'use_dense_net':True, 
              'use_context_net':True,
              'use_atlas':True,
              'use_affine':True,
              'use_def':True,
              'batch_size':10,
              'depth':256,
              'height':256, 
              'width':256,
              'gamma': 1e-5,
              'cost_search_range':2,
              'lowest': 4,
              'last':1}
    model,_,_ = create_model(config = config, name="Model")
    print(model.summary())
