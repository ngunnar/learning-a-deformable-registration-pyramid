from pathlib import Path
import nibabel as nib
from skimage.transform import resize
import tensorflow as tf
import numpy as np

class TensorflowDatasetLoader:   
    def __init__(self, idxs, depth, height, width, batch_size, lowest, last, use_label):
        self.idxs = idxs
        self.crop_img = False

        self.input_dim = (depth,height,width)
        output_shape = []
        output_types = []
        for i in range(1, lowest - last + 2):
            l = lowest + 1 - i
            x_dim = depth // (2**l)
            y_dim = height // (2**l)
            z_dim = width // (2**l)
            output_shape.append((x_dim, y_dim, z_dim,3))
            output_shape.append((x_dim, y_dim, z_dim,2))
            output_types.append(tf.float32)
            output_types.append(tf.float32)

        output_shape.append((*self.input_dim,3))
        output_shape.append((*self.input_dim,1))
        output_shape.append((*self.input_dim,1))
        output_types.append(tf.float32)
        output_types.append(tf.float32)
        output_types.append(tf.float32)

        input_data = tf.data.Dataset.from_generator(
            self.generator(self.input_dim, lowest, last, idxs, True),
            (tf.float32, tf.float32, tf.float32),
            ((*self.input_dim,1),(*self.input_dim,1),(*self.input_dim,1)))

        output_data = tf.data.Dataset.from_generator(
            self.generator(self.input_dim, lowest, last, idxs, False),
            tuple(output_types), 
            output_shapes = tuple(output_shape))

        dataset = tf.data.Dataset.zip((input_data, output_data))
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=50)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = dataset 

    def _get_train_samples(self, idx):
        images, labels, out = [], [], []
        use_atlas = True
        crop_img = False
        if 'task_04' in idx[0]:
            crop_img=True
            c = 6
        
        fixed = nib.load(idx[0]).get_fdata()

        fixed = TensorflowDatasetLoader.normalize(fixed)            
        assert not np.any(np.isnan(fixed)), idx[0]
        moving = nib.load(idx[1]).get_fdata()

        moving = TensorflowDatasetLoader.normalize(moving)

        assert not np.any(np.isnan(moving)), idx[1]
        if idx[2] is not None:
            fixed_label = nib.load(idx[2]).get_fdata()
            assert not np.any(np.isnan(fixed_label)), l[2]
            moving_label = nib.load(idx[3]).get_fdata()
            assert not np.any(np.isnan(moving_label)), l[3]
            if crop_img:
                fixed_label = TensorflowDatasetLoader.crop(fixed_label, c)
                moving_label = TensorflowDatasetLoader.crop(moving_label, c)
            fixed_label = resize(fixed_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
            moving_label = resize(moving_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        else:
            use_atlas = False
            fixed_label, moving_label = None, None

        if crop_img:
            fixed = TensorflowDatasetLoader.crop(fixed, c)
            moving = TensorflowDatasetLoader.crop(moving, c)
        fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        moving = resize(moving, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')

        images.append([fixed, moving])
        images = [fixed, moving]    
        if use_atlas:
            labels = [fixed_label, moving_label]
            out = images, labels
        else:
            out = images, None
        return out
    
    def _get_input_samples(self, idx):
        use_atlas = True
        crop_img = False
        if 'task_04' in idx[0]:
            crop_img=True
            c = 6
            
        fixed = nib.load(idx[0]).get_fdata()

        fixed = TensorflowDatasetLoader.normalize(fixed)            
        assert not np.any(np.isnan(fixed)), idx[0]
        moving = nib.load(idx[1]).get_fdata()

        moving = TensorflowDatasetLoader.normalize(moving)

        assert not np.any(np.isnan(moving)), idx[1]
        if idx[3] is not None:
            moving_label = nib.load(idx[3]).get_fdata()
            assert not np.any(np.isnan(moving_label)), l[3]
            if crop_img:
                moving_label = TensorflowDatasetLoader.crop(moving_label, c)
            moving_label = resize(moving_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        else:
            use_atlas = False
            moving_label = None

        if crop_img:
            fixed = TensorflowDatasetLoader.crop(fixed, c)
            moving = TensorflowDatasetLoader.crop(moving, c)
        fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        moving = resize(moving, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')

        images = [fixed, moving]    
        if use_atlas:
            out = images, moving_label
        else:
            out = images, None
        return out

    def _get_output_samples(self, idx):
        crop_img = False
        use_atlas = True
        if 'task_04' in idx[0]:
            crop_img=True
            c = 6
        
        fixed = nib.load(idx[0]).get_fdata()

        fixed = TensorflowDatasetLoader.normalize(fixed)            
        assert not np.any(np.isnan(fixed)), idx[0]
        
        if idx[2] is not None:
            fixed_label = nib.load(idx[2]).get_fdata()
            assert not np.any(np.isnan(fixed_label)), l[2]
            if crop_img:
                fixed_label = TensorflowDatasetLoader.crop(fixed_label, c)
            fixed_label = resize(fixed_label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        else:
            use_atlas = False
            fixed_label = None

        if crop_img:
            fixed = TensorflowDatasetLoader.crop(fixed, c)
        fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')

        if use_atlas:
            out = fixed, fixed_label
        else:
            out = fixed, None
        return out
    
    def generator(self, input_dim, lowest, last, idxs, use_input = True):
        def gen():
            for i in range(len(idxs)):
                if use_input:
                    images, moving_label = self._get_input_samples(idx=idxs[i])
                    inputs = [images[0], images[1]]
                
                    if moving_label is not None:
                        inputs.append(moving_label)
                    
                    yield tuple(inputs)
                else:
                    fixed, fixed_label = self._get_output_samples(idx=idxs[i])
                    out = []
                    zeros = np.zeros((*input_dim, 3), dtype='float32')
                    l = 0
                    for i in range(1, lowest - last + 2):
                        l = lowest + 1 - i
                        out_flow = resize(zeros[...,0], tuple([x//(2**l) for x in input_dim]), mode='constant')[...,None]
                        out_flow1 = np.repeat(out_flow, 3, axis=-1)
                        out_flow2 = np.repeat(out_flow, 2, axis=-1)
                        out.append(out_flow1)
                        #DUMMY
                        out.append(out_flow2)
                    out.append(zeros)
                    out.append(fixed)
                    if fixed_label is not None:
                        out.append(fixed_label)
                    yield tuple(out)                
        return gen

    def normalize(arr):
        arr_min = np.min(arr)
        return (arr-arr_min)/(np.max(arr)-arr_min)

    def crop(img, c):
        img = img[img.shape[0]//c:(c-1)*img.shape[0]//c,
                  img.shape[1]//c:(c-1)*img.shape[1]//c,
                  img.shape[2]//c:(c-1)*img.shape[2]//c]
        return img
