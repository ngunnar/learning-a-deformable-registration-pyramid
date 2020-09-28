from pathlib import Path
import nibabel as nib
from skimage.transform import resize
from PIL import ImageOps
import tensorflow as tf
import numpy as np
from scipy import ndimage

# TODO fix handling of use_label

class TensorflowDatasetLoader:   
    def __init__(self, idxs, depth, height, width, batch_size, lowest, last, use_label):
        self.use_label = use_label        
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
            output_shape.append((x_dim, y_dim, z_dim,3)) # affine
            output_shape.append((x_dim, y_dim, z_dim,3)) # deformable
            output_shape.append((x_dim, y_dim, z_dim,2)) # image sim placeholder
            output_types.append(tf.float32)
            output_types.append(tf.float32)
            output_types.append(tf.float32)

        output_shape.append((*self.input_dim,3)) # affine
        output_types.append(tf.float32)
        output_shape.append((*self.input_dim,3)) # deformable
        output_types.append(tf.float32)
        
        output_shape.append((*self.input_dim,1)) # image sim
        output_types.append(tf.float32)
        
        if use_label:
            output_shape.append((*self.input_dim,1)) # seg
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
       
    def _get_org_data(self, idx):
        f_f, fl_f, f_r, f_r_ax, m_f, ml_f, m_r, m_r_ax, flip_axes, crop_img, task = TensorflowDatasetLoader.parse_idx(idx)
        fixed = nib.load(f_f).get_fdata()
        moving = nib.load(m_f).get_fdata()
        if fl_f is not None and ml_f is not None:
            fixed_label = nib.load(fl_f).get_fdata().astype('int')
            moving_label = nib.load(ml_f).get_fdata().astype('int')
        else:
            fixed_label, moving_label = None, None
        return [fixed, moving], [fixed_label, moving_label]
        
    def _get_train_samples(self, idx):
        use_atlas = True
        
        f_f, fl_f, f_r, f_r_ax, m_f, ml_f, m_r, m_r_ax, flip_axes, crop_img, task = TensorflowDatasetLoader.parse_idx(idx)
        fixed = nib.load(f_f).get_fdata()
        moving = nib.load(m_f).get_fdata()
        
        if crop_img is not None:
            fixed = TensorflowDatasetLoader.crop(fixed, crop_img)
            moving = TensorflowDatasetLoader.crop(moving, crop_img)
        
        fixed = TensorflowDatasetLoader.img_augmentation(fixed, f_r, f_r_ax, flip_axes)
        moving = TensorflowDatasetLoader.img_augmentation(moving, m_r, m_r_ax, flip_axes)
        
        if task in [2,3]:
            fixed, moving = TensorflowDatasetLoader.normalize(fixed, moving)          
        else:
            fixed = TensorflowDatasetLoader.normalize(fixed)
            moving = TensorflowDatasetLoader.normalize(moving)         
        
        fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        moving = resize(moving, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32') 
        
        assert not np.any(np.isnan(fixed)), f_f
        assert not np.any(np.isnan(moving)), m_f
        
        if fl_f is not None and ml_f is not None:
            fixed_label = self._get_label(fl_f, crop_img, f_r, f_r_ax, flip_axes)
            moving_label = self._get_label(ml_f, crop_img, m_r, m_r_ax, flip_axes)
        else:
            use_atlas = False
            fixed_label, moving_label = None, None
        
        images = [fixed, moving]    
        if use_atlas:
            labels = [fixed_label, moving_label]
            out = images, labels
        else:
            out = images, None
        return out
    
    def _get_label(self, file, crop_img, rot_angle, rot_ax, flip_axes):
        label = nib.load(file).get_fdata().astype('int')
        if crop_img is not None:
            label = TensorflowDatasetLoader.crop(label, crop_img)
        label = TensorflowDatasetLoader.img_augmentation(label, rot_angle, rot_ax, flip_axes)
        assert not np.any(np.isnan(label)), file
            
        label = resize(label, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        return label
    
    def _get_input_samples(self, idx):
        use_atlas = True
        f_f, _, f_r, f_r_ax, m_f, ml_f, m_r, m_r_ax, flip_axes, crop_img, task = TensorflowDatasetLoader.parse_idx(idx)
        
        fixed = nib.load(f_f).get_fdata()
        moving = nib.load(m_f).get_fdata()
        
        if crop_img is not None:
            fixed = TensorflowDatasetLoader.crop(fixed, crop_img)
            moving = TensorflowDatasetLoader.crop(moving, crop_img)
        
        fixed = TensorflowDatasetLoader.img_augmentation(fixed, f_r, f_r_ax, flip_axes)
        moving = TensorflowDatasetLoader.img_augmentation(moving, m_r, m_r_ax, flip_axes)
        
        if task in [2,3]:
            fixed, moving = TensorflowDatasetLoader.normalize(fixed, moving)          
        else:
            fixed = TensorflowDatasetLoader.normalize(fixed)
            moving = TensorflowDatasetLoader.normalize(moving)
        
        fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        moving = resize(moving, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        
        assert not np.any(np.isnan(fixed)), f_f       
        assert not np.any(np.isnan(moving)), m_f
        
        if ml_f is not None:
            moving_label = self._get_label(ml_f, crop_img, m_r, m_r_ax, flip_axes)            
        else:
            use_atlas = False
            moving_label = None    
        
        images = [fixed, moving]    
        if use_atlas:
            out = images, moving_label
        else:
            out = images, None
        return out

    def _get_output_samples(self, idx):
        use_atlas = True
        
        f_f, fl_f, f_r, f_r_ax, _, _, _, _, flip_axes, crop_img, task = TensorflowDatasetLoader.parse_idx(idx)
        
        fixed = nib.load(f_f).get_fdata()
        
        if crop_img is not None:
            fixed = TensorflowDatasetLoader.crop(fixed, crop_img)
            
        fixed = TensorflowDatasetLoader.img_augmentation(fixed, f_r, f_r_ax, flip_axes)
        fixed = TensorflowDatasetLoader.normalize(fixed)            
        fixed = resize(fixed, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        assert not np.any(np.isnan(fixed)), f_f
        
        if fl_f is not None:
            fixed_label = self._get_label(fl_f, crop_img, f_r, f_r_ax, flip_axes)
        else:
            use_atlas = False
            fixed_label = None
        
        if use_atlas:
            out = fixed, fixed_label
        else:
            out = fixed, None
        return out
    
    def generator(self, input_dim, lowest, last, idxs, use_input = True):
        def gen():
            for i in range(len(idxs)):
                if use_input:
                    images, moving_label = self._get_input_samples(idxs[i])
                    inputs = [images[0], images[1]]
                
                    if moving_label is not None:
                        inputs.append(moving_label)
                    
                    yield tuple(inputs)
                else:
                    fixed, fixed_label = self._get_output_samples(idxs[i])
                    out = []
                    zeros = np.zeros((*input_dim, 3), dtype='float32')
                    l = 0
                    for i in range(1, lowest - last + 2):
                        l = lowest + 1 - i
                        out_flow = resize(zeros[...,0], tuple([x//(2**l) for x in input_dim]), mode='constant')[...,None]
                        out_flow1 = np.repeat(out_flow, 3, axis=-1)
                        out_flow2 = np.repeat(out_flow, 2, axis=-1)
                        out.append(out_flow1) # Affine
                        out.append(out_flow1) # Deformable
                        #DUMMY
                        out.append(out_flow2) # Placeholder for images sim
                    
                    out.append(zeros) # Affine
                    out.append(zeros) # Deformable
                    out.append(fixed) # Image sim
                    if fixed_label is not None:
                        out.append(fixed_label) # Seg
                    yield tuple(out)                
        return gen
    
    def parse_idx(idx):
        fixed = idx[0][0]
        fixed_label = idx[0][1]
        fixed_rot = idx[0][2][0]
        fixed_rot_ax = idx[0][2][1]
        
        moving = idx[1][0]
        moving_label = idx[1][1]
        moving_rot = idx[1][2][0]
        moving_rot_ax = idx[1][2][1]
        
        flip_axes = idx[2]
        crop_img = idx[3]
        task = idx[4]
        return fixed, fixed_label, fixed_rot, fixed_rot_ax, moving, moving_label, moving_rot, moving_rot_ax, flip_axes, crop_img, task
    
    def img_augmentation(img, rot_angle, rot_ax, flip_axes):
        if flip_axes > -1:
            img = np.flip(img, flip_axes)
        if rot_angle != 0:
            img = ndimage.rotate(img, rot_angle, axes=(rot_ax[0], rot_ax[1]), mode='nearest', reshape=False)
        return img
    
    def normalize(img1, img2 = None):
        img1_max = np.max(img1)
        img1_min = np.min(img1)
        out_1 = (img1-img1_min)/(img1_max-img1_min)
        if img2 is not None:
            out_2 = (img2-img1_min)/(img1_max-img1_min)
            return out_1, out_2
        return out_1

    def crop(img, c):
        img = img[img.shape[0]//c:(c-1)*img.shape[0]//c,
                  img.shape[1]//c:(c-1)*img.shape[1]//c,
                  img.shape[2]//c:(c-1)*img.shape[2]//c]
        return img
