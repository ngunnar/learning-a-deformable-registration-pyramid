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
from CustomLayers.AffineLayer import Affine
from CustomLayers.TransformAffineFlowLayer import TransformAffineFlow
from tensorflow.keras.layers import Concatenate
import losses

from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from neuron.layers import VecInt, Negate

import tensorflow as tf
import numpy as np

def create_model(config, name):
    
    def w_loss(loss):
        def l(_, yp):
            i = yp[..., :yp.shape[-1]//2]
            w = yp[..., yp.shape[-1]//2:]
            return loss(i, w)
        return l
    
    useDenseNet = config['use_dense_net']
    useContextNet = config['use_context_net']
    #batch_size = None#config['batch_size']
    depth = config['depth']
    height = config['height']
    width = config['width']
    gamma = config['gamma']
    lowest = config['lowest']
    last = config['last']
    cost_search_range = config['cost_search_range']
    use_atlas = config['use_atlas']
    use_affine = config['use_affine']
    use_def = config['use_def']
    
    d_l = config['data_loss']
    assert d_l in ['mse', 'cc', 'ncc', 'mse_ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    
    if d_l in ['ncc', 'cc']:
        d_l = losses.NCC().loss
    elif d_l == 'mse_ncc':
        d_l = losses.NCC().loss_with_mse
    else:
        #d_l = tf.keras.losses.MeanSquaredError(reduction='none')
        d_l = tf.keras.losses.MeanSquaredError()
    
    ## Initiazing Layers
    pyramid = Pyramid(gamma=gamma, num = lowest)
    cost = CostVolume(search_range=cost_search_range)
    
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
        flow_ests.append(OpticalFlowEstimator(i=i, gamma=gamma, denseNet= useDenseNet, first = first))
        if use_affine:
            affines.append(Affine(i=i))
        if i != last:
            ups.append(Upsampling(i=i, gamma= gamma))
        
        if useContextNet or i == last:
            contexs.append(Context(i=i, gamma=gamma))

    ### Creating model ##
    fixed = tf.keras.layers.Input(shape=(depth, height, width, 1), name='fixed_img')
    moving = tf.keras.layers.Input(shape=(depth, height, width, 1), name='moving_img')
    batch_size = tf.shape(fixed, name='batch_size')[0]
    
    inputs = [fixed, moving]
    if use_atlas:
        moving_seg = tf.keras.layers.Input(shape=(depth, height, width, 1), name='moving_seg')
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
            a_flow = affines[l]([out1[l], warp])
            flow = TransformAffineFlow(shape)([a_flow, flow])
            warp = warps_2[l]([warp, flow])
            flow = Lambda(lambda x:x, name = "est_aff_flow{0}".format(l))(flow)
            outputs.append(flow)
            loss.append(losses.Grad('l2').loss)
            loss_weights.append(config['alphas'][l])
        
        if use_def or l == 0:
            cv = cost([out1[l], warp])
        
            if i != 1 and use_def:
                [upfeat, flow] = flow_ests[l]([cv, out1[l], flow, up_feat])
            else:
                [upfeat, flow] = flow_ests[l]([cv, out1[l], flow])       
                
            if useContextNet:
                flow = contexs[l]([upfeat, flow])
            elif i == last:
                flow = contexs[0]([upfeat, flow])     
        
        flow = Lambda(lambda x:x, name = "est_flow{0}".format(l))(flow)
        
        if l != 0:
            outputs.append(flow)
            loss.append(losses.Grad('l2').loss)
            loss_weights.append(config['betas'][l])
            r = Resize(scalar = 1.0, factor = 1/(2**l), name='p_score_{0}'.format(l))
            warp_moving = Warp(name='warp_m_{0}'.format(i+1))([r(moving), flow])
            #o = Concatenate(axis=-1, name='sim_{0}'.format(l))([out1[l], warp])
            o = Concatenate(axis=-1, name='sim_{0}'.format(l))([r(fixed), warp_moving])
            outputs.append(o)
            loss.append(w_loss(d_l))
            loss_weights.append(config['reg_params'][l])
        if l != 0:
            if use_def:
                up_flow, up_feat = ups[l-1]([flow, upfeat])
            else:
                up_flow = ups[l-1]([flow])
    
    flow_est = flow
    flow_int = VecInt(method='ss', name='flow_int', int_steps=7)(flow_est)
    #flow_neg = Negate()(flow_est)
    #flow_neg = VecInt(method='ss', name='neg_flow-int', int_steps=7)(flow_neg)
    
    warped = Warp(name='sim')([moving, flow_est])    
    outputs.append(flow_est)
    loss.append(losses.Grad('l2').loss)
    loss_weights.append(config['betas'][0])
    
    outputs.append(warped)
    loss.append(d_l)
    loss_weights.append(config['reg_params'][0])
    
    if use_atlas:
        warped_moving_seg = Warp(name='seg')([moving_seg, flow_int])
        outputs.append(warped_moving_seg)
        loss.append(losses.Dice().loss)
        loss_weights.append(config['atlas_wt'])
    
    return Model(inputs=inputs, outputs=outputs, name=name), loss, loss_weights


if __name__ == "__main__":
    config = {'use_dense_net':True, 'use_context_net':True, 'batch_size':10, 'depth':256, 'height':256, 'width':256, 'gamma': 1e-5, 'lowest': 4, 'last':1}
    pwc_model = create_model(config = config, name="PWC_Net")
    print(pwc_model.summary())
