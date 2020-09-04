import tensorflow as tf
import numpy as np
import math
import nibabel as nib
from skimage.transform import resize
import copy
from scipy import ndimage

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, idxs, depth, heigth, width, batch_size, lowest, last, shuffle = True):                
        self.idxs = idxs        
        self.input_dim = (depth, heigth, width)
        self.batch_size = batch_size
        self.lowest = lowest
        self.last = last
        self.shuffle = shuffle
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
        
        #assert images.shape[1] == 2, images.shape
        #assert images.shape[2] == self.input_dim[0], images.shape
        #assert images.shape[3] == self.input_dim[1], images.shape
        #assert images.shape[4] == self.input_dim[2], images.shape
        #assert images.shape[5] == 1, images.shape
        
        fixed = images[:,0,:,:]
        moving = images[:,1,:,:]
        if labels is not None:
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
            out.append(tf.convert_to_tensor(out_flow))
            #DUMMY
            out.append(tf.convert_to_tensor(out_flow))
        
        out.append(tf.convert_to_tensor(zeros))
        out.append(fixed)
        inputs = [fixed, moving]
        if labels is not None:
            out.append(fixed_seg)        
            inputs.append(moving_seg)
            
        return (inputs, out)

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
    
    def _get_train_samples(self, idx):
        images, labels, out = [], [], []
        use_atlas = True
        for l in idx:
            crop_img = False
            if 'task_04' in l[0]:
                crop_img=True
                c = 6
            fixed = nib.load(l[0]).get_data()
            
            fixed = CustomDataGenerator.normalize(fixed)            
            assert not np.any(np.isnan(fixed)), l[0]
            moving = nib.load(l[1]).get_data()
            
            moving = CustomDataGenerator.normalize(moving)
            
            assert not np.any(np.isnan(moving)), l[1]
            use_atlas = True
            if l[2] is not None:
                fixed_label = nib.load(l[2]).get_data()
                assert not np.any(np.isnan(fixed_label)), l[2]
                moving_label = nib.load(l[3]).get_data()
                if crop_img:
                    fixed_label = CustomDataGenerator.crop(fixed_label, c)
                    moving_label = CustomDataGenerator.crop(moving_label, c)
                fixed_label = resize(fixed_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('int')
                moving_label = resize(moving_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('int')
            else:
                use_atlas = False
                fixed_label, moving_label = None, None
            
            if crop_img:
                fixed = CustomDataGenerator.crop(fixed, c)
                moving = CustomDataGenerator.crop(moving, c)
            fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None]
            moving = resize(moving, self.input_dim, preserve_range=True, mode='constant')[...,None]
            
            images.append([fixed, moving])
            #if use_atlas:
            labels.append([fixed_label, moving_label])
        
        if use_atlas:
            out = tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.float32)
        else:
            out = tf.convert_to_tensor(images, dtype=tf.float32), None
        return out

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.idxs)