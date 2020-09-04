import tensorflow as tf
import numpy as np
import math
import nibabel as nib
from skimage.transform import resize
import copy
from scipy import ndimage
from debugUtil import sphere, random_deformation_linear
import neuron.layers as nrn_layers

class DebugDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, depth, height, width, batch_size, lowest, last):
        self.idxs = ['task_00']
        self.input_dim = (depth, height, width)
        zero = 0.5
        first = 0.3
        second = 0.8
        third = 0.6
        diff = 0.1
        
        fixed_0 = sphere((depth, height, width), depth//3, (depth//2, height//2, width//2)).astype('float32')        
        fixed_1 = sphere((depth, height, width), depth//5, (depth//2, height//2, width//2)).astype('float32')
        t_1 = ((fixed_1 > 1.0-diff) & (fixed_1 < 1.0+diff)).astype('float32')
        fixed_2 = sphere((depth, height, width), depth//6, (depth//4, height//4, width//2)).astype('float32')
        t_2 = ((fixed_2 > 1.0-diff) & (fixed_2 < 1.0+diff)).astype('float32')
        fixed_3 = sphere((depth, height, width), depth//5+5, (depth//2, height//2, width//2)).astype('float32')
        fixed_3 = fixed_3 - fixed_1
        t_3 = ((fixed_3 > 1.0-diff) & (fixed_3 < 1.0+diff)).astype('float32')
        
        
        fixed = fixed_0*zero + fixed_1*first + fixed_2*second + fixed_3*third + np.random.normal(0, 0.05, (depth, height, width))
        
        fixed_label = t_1
        fixed_label += t_2*2
        fixed_label += t_3*3
        fixed = fixed[...,None]
        fixed_label = fixed_label[...,None]
        
        flow = random_deformation_linear([1,depth,height,width], 10, 10)
        warp = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij',single_transform=False)
        
        moving = warp([fixed[None,...], flow]).numpy()[0,...]
        moving_label = warp([fixed_label[None,...], nrn_layers.VecInt(method='ss', name='flow_int', int_steps=7)(flow)])[0,...]
        
        self.images = tf.convert_to_tensor([fixed, moving], dtype=tf.float32)[None,...]
        self.labels = tf.convert_to_tensor([fixed_label, moving_label], dtype=tf.float32)[None,...]
        
        self.batch_size = batch_size
        self.lowest = lowest
        self.last = last
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.idxs) / self.batch_size)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        images, labels = self.images, self.labels     
        
        #assert tf.math.reduce_max(images) <= 1, tf.math.reduce_max(images)
        #assert tf.math.reduce_max(images) >= 0, tf.math.reduce_max(images)
        
        #assert images.shape[1] == 2, images.shape
        #assert images.shape[2] == self.input_dim[0], images.shape
        #assert images.shape[3] == self.input_dim[1], images.shape
        #assert images.shape[4] == self.input_dim[2], images.shape
        #assert images.shape[5] == 1, images.shape
        
        fixed = images[:,0,...]
        moving = images[:,1,...]
        if labels is not None:
            moving_seg = labels[:,1,...]
            fixed_seg = labels[:,0,...]
        
        volshape = fixed.shape[1:-1]
        zeros = np.zeros((self.batch_size, *volshape, 3))
        inp, out = [], []
        l = 0
        for i in range(1, self.lowest - self.last + 2):
            out_flow = resize(zeros[0,:,:,:,0], tuple([x//(2**i) for x in self.input_dim]), mode='constant')[None,:,:,:,None]
            out_flow = np.repeat(out_flow, 3, axis=-1)
            out_flow = np.repeat(out_flow, self.batch_size, axis=0)
            out.append(tf.convert_to_tensor(out_flow))
            #DUMMY
            out.append(tf.convert_to_tensor(out_flow))
        
        out.append(tf.convert_to_tensor(zeros))
        out.append(fixed)
        inputs = [fixed, moving]
        if labels is not None:
            #out.append(moving_seg)        
            #inputs.append(fixed_seg)
            out.append(fixed_seg)        
            inputs.append(moving_seg)
        #else:
            
        return (inputs, out)
    
    def _get_train_samples(self, idx):
        return self.images, self.labels
    def normalize(arr):
        arr_min = np.min(arr)
        return (arr-arr_min)/(np.max(arr)-arr_min)
    
    def scale_by(arr, fac):
        mean = np.mean(arr)
        return (arr-mean)*fac + mean
    
    def grad(image):
        # Get x-gradient in "sx"
        sx = ndimage.sobel(image,axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(image,axis=1,mode='constant')
        # Get z-gradient in "sz"
        sz = ndimage.sobel(image,axis=2,mode='constant')
        # Get square root of sum of squares
        image=np.hypot(sx,sy,sz)
        return image
    
    def crop(img, c):
        img = img[img.shape[0]//c:(c-1)*img.shape[0]//c,
              img.shape[1]//c:(c-1)*img.shape[1]//c,
              img.shape[2]//c:(c-1)*img.shape[2]//c]
        return img

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
