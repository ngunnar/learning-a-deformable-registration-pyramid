import tensorflow as tf
import numpy as np
import math
import nibabel as nib
from Utils.tensorflow_addons import resize3D
from skimage.transform import resize
import copy
from scipy import ndimage

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, idxs, depth, heigth, width, batch_size, lowest, last, use_atlas, shuffle = True):                
        self.idxs = idxs        
        self.input_dim = (depth, heigth, width)
        self.batch_size = batch_size
        self.lowest = lowest
        self.last = last
        self.shuffle = shuffle
        self.use_atlas = use_atlas
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.idxs) / self.batch_size)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.idxs[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        images, labels = self._get_train_samples(idx=indexes)      
        
        #assert tf.math.reduce_max(images) <= 1, tf.math.reduce_max(images)
        #assert tf.math.reduce_max(images) >= 0, tf.math.reduce_max(images)
        
        assert images.shape[1] == 2, images.shape
        assert images.shape[2] == self.input_dim[0], images.shape
        assert images.shape[3] == self.input_dim[1], images.shape
        assert images.shape[4] == self.input_dim[2], images.shape
        assert images.shape[5] == 1, images.shape
        
        fixed = images[:,0,:,:]
        moving = images[:,1,:,:]
        if self.use_atlas:
            moving_seg = labels[:,1,:,:]
            fixed_seg = labels[:,0,:,:]
        
        volshape = fixed.shape[1:-1]
        zeros = np.zeros((self.batch_size, *volshape, 3))
        inp, out = [], []
        l = 0
        for i in range(1, self.lowest - self.last + 2):
            out_flow = resize(zeros[0,:,:,:,0], tuple([x//(2**i) for x in self.input_dim]), mode='constant')[None,:,:,:,None]
            out_flow = np.repeat(out_flow, 3, axis=-1)
            out_flow = np.repeat(out_flow, self.batch_size, axis=0)
            out.append(out_flow)
            #DUMMY
            out.append(out_flow)
        
        out.append(zeros)
        out.append(fixed)
        inputs = [fixed, moving]
        if self.use_atlas:
            out.append(moving_seg)        
            inputs.append(fixed_seg)
        return (inputs, out)
    
    '''
    def normalize(img):
        _min = -3024.0005
        _max = 3071.0007
        #_min = tf.math.reduce_min(img)
        #_max = tf.math.reduce_max(img)
        return (img - _min) / (_max - _min)
    '''
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
    
    def _get_train_samples(self, idx):
        images, labels, out = [], [], []
        for l in idx:
            fixed = nib.load(l[0]).get_data()
            #fixed = CustomDataGenerator.grad(fixed)
            
            #fixed = np.clip(CustomDataGenerator.scale_by(np.clip(CustomDataGenerator.normalize(fixed)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
            fixed = CustomDataGenerator.normalize(fixed)
            #fixed[fixed < 0.55] = 0
            
            #fixed = CustomDataGenerator.normalize(fixed)
            assert not np.any(np.isnan(fixed)), l[0]
            moving = nib.load(l[1]).get_data()
            
            #moving = CustomDataGenerator.grad(moving)
            
            #moving = np.clip(CustomDataGenerator.scale_by(np.clip(CustomDataGenerator.normalize(moving)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
            moving = CustomDataGenerator.normalize(moving)
            #moving[moving < 0.55] = 0    
            #moving = CustomDataGenerator.normalize(moving)
            
            assert not np.any(np.isnan(moving)), l[1]
            if self.use_atlas:
                fixed_label = nib.load(l[2]).get_data()
                assert not np.any(np.isnan(fixed_label)), l[2]
                moving_label = nib.load(l[3]).get_data()
                fixed_label = resize(fixed_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('int')
                moving_label = resize(moving_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('int')
            else:
                fixed_label, moving_label = None, None


            
            fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None]
            moving = resize(moving, self.input_dim, preserve_range=True, mode='constant')[...,None]
            
            #fixed = resize3D(fixed[None,:,:,:,None], self.input_dim)[0,...]
            #moving = resize3D(moving[None,:,:,:,None], self.input_dim)[0,...]
            #fixed_label = resize3D(fixed_label[None,:,:,:,None], self.input_dim)[0,...]
            #moving_label = resize3D(moving_label[None,:,:,:,None], self.input_dim)[0,...]
            
            images.append([fixed, moving])
            if self.use_atlas:
                labels.append([fixed_label, moving_label])
        
        if self.use_atlas:
            out = tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.float32)
        else:
            out = tf.convert_to_tensor(images, dtype=tf.float32), None
        return out

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.idxs)