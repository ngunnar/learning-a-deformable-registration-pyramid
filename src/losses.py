import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor, EagerTensor

def dice_loss(y_true,
              y_pred,
              classes,
              smooth=1e-5,
              include_background=False,
              only_present=True):
    """Calculates a smooth Dice coefficient loss from sparse labels.
    Args:
        logits (tf.Tensor): logits prediction for which to calculate
            crossentropy error
        labels (tf.Tensor): sparse labels used for crossentropy error
            calculation
        num_classes (int): number of class labels to evaluate on
        smooth (float): smoothing coefficient for the loss computation
        include_background (bool): flag to include a loss on the background
            label or not
        only_present (bool): flag to include only labels present in the
            inputs or not
    Returns:
        tf.Tensor: Tensor scalar representing the loss
    """
    #print(y_true.dtype)
    #y_true = tf.one_hot(tf.cast(y_true, tf.int32), classes)
    #y_pred = tf.nn.softmax(y_pred)
    
    #remove_classes = tf.reduce_sum(
    y_true_sum = tf.reduce_sum(y_true, axis=[1, 2, 3], name='label_sum')
    y_pred_sum = tf.reduce_sum(y_pred, axis=[1, 2, 3], name='pred_sum')
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3],
                                 name='intersection')
    
    per_sample_per_class_dice = (2. * intersection + 1e-5)
    per_sample_per_class_dice /= (y_true_sum + y_pred_sum + 1e-5)
    
    
    if include_background: #include background
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice, (-1, ))
        flat_label = tf.reshape(y_true_sum, (-1, ))
    else:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice[:, 1:], (-1, ))
        flat_label = tf.reshape(y_true_sum[:, 1:], (-1, ))

    # Include or exclude non-present labels for the computation
    if only_present:
        masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                      tf.logical_not(tf.equal(flat_label, 0)))
    else:
        masked_dice = tf.boolean_mask(
            flat_per_sample_per_class_dice,
            tf.logical_not(tf.math.is_nan(flat_per_sample_per_class_dice)))
        

    dice = tf.reduce_mean(masked_dice)
    loss = 1. - dice

    return loss

def mse_loss(y_true, y_pred):
    #mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

def dice_coe(y_true, y_pred, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    classes = tf.cast(tf.math.reduce_max(y_true), tf.int32)    
    y_true = tf.cast(tf.math.round(y_true), tf.int32)
    
    inse = tf.reduce_sum(y_true * y_pred, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(y_true * y_true, axis=axis)
        r = tf.reduce_sum(y_pred * y_pred, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(y_true, axis=axis)
        r = tf.reduce_sum(y_pred, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

def dice_coef_binary(y_true, y_pred, num_classes=2, smooth=1e-7):
    '''
    Dice coefficient for X categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[...,1:])
    y_pred_f = K.flatten(K.one_hot(K.cast(y_pred, 'int32'), num_classes=num_classes)[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


class Dice():
    """
    Dice binary loss
    """
    def __init__(self, label_classes):
        self.label_classes = label_classes

    def loss(self, y_true, y_pred):
        return dice_loss(y_true, y_pred, self.label_classes)
    
class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, last_dim = 1, win=None, eps=1e-5):
        self.win = win
        self.eps = eps
        self.last_dim = last_dim

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        
        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
        
        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        self.last_dim = J.shape[-1]
        sum_filt = tf.ones([*self.win, self.last_dim, self.last_dim])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)
    
    def loss_with_mse(self, I, J):
        return self.loss(I,J) + tf.keras.losses.mean_squared_error(I,J)

    
class Affine_loss():
    def __init__(self):
        self.A_I = np.array([[1,0,0,0,0,1,0,0,0,0,1,0]], 'float32')
        self.A_I = tf.convert_to_tensor(self.A_I)
    def loss(self, _, y_pred):
        #print(y_pred.shape, self.A_I.shape)
        #return tf.nn.l2_loss(y_pred - self.A_I)
        return tf.keras.losses.mean_squared_error(y_pred,self.A_I)
    
    
class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)