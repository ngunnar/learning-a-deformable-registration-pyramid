from create_model import create_model
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = {
        'depth': 64,
        'height': 64,
        'width': 64,
        'batch_size': 1,
        'use_affine': True,
        'use_def': True,
        'use_dense_net': True,    
        'use_context_net': False,
        'gamma':0.0,
        'weights': None,
        'use_atlas': False,
        'cost_search_range': 2,
        'lowest':4,
        'last':1,
        'reg_params': [1.0, 0.1, 0.05, 0.02, 0.01],
        'seg_loss': 'dice',
        'data_loss': 'ncc',
        'betas': [1.0, 0.25, 0.05, 0.0125, 0.002],
        'alphas': [1.0, 0.25, 0.05, 0.0125, 0.002],
        'atlas_wt':1.0,
    }

class Model():

    def __init__(self, task_type):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        self.config = config
        assert task_type in [1,2,3,4]
        self.task_type = task_type
        
        if task_type == 1:
            self.config['weights'] = None
        elif task_type == 2:
            self.config['weights'] = './Models/task2_model'
        elif task_type == 3:
            self.config['weights'] = './Models/task3_model'
        else:
            self.config['weights'] = './Models/task4_model'
            
        self.input_dim = (self.config['depth'],self.config['height'],self.config['width'])
        self.model, _, _ = create_model(config = self.config, name="Model")
        if self.config['weights'] is not None:
            self.model.load_weights(config['weights']).expect_partial()
    
    def normalize(self, img1, img2 = None):
        img1_max = np.max(img1)
        img1_min = np.min(img1)
        out_1 = (img1-img1_min)/(img1_max-img1_min)
        if img2 is not None:
            out_2 = (img2-img1_min)/(img1_max-img1_min)
            return out_1, out_2
        return out_1
    
    def predict(self, fixed, moving):
        t0 = time.time()
        print("Loading images...")
        if self.task_type in [2,3]:
            fixed_prep, moving_prep = self.normalize(fixed, moving)
        else:
            fixed_prep = self.normalize(fixed)
            moving_prep = self.normalize(moving)
        fixed_prep = resize(fixed_prep, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        moving_prep = resize(moving_prep, self.input_dim, preserve_range=True, mode='constant')[...,None].astype('float32')
        t0 = time.time()
        disp_field = self.model.predict([fixed_prep[None,...], moving_prep[None,...]])[-2]
        t1 = time.time()
        disp = resize(disp_field[0,...]/self.input_dim[0], fixed.shape) # TODO
        disp[...,0] *= fixed.shape[0]
        disp[...,1] *= fixed.shape[1]
        disp[...,2] *= fixed.shape[2]
        #assert len(disp.shape) == 4, disp.shape
        out = np.moveaxis(disp, -1, 0)
        print("Prediction done ({:.2f}s)!".format(t1-t0))
        return out, t1 - t0
