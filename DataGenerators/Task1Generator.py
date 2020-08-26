import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator

valid_items = ['1']

class Task1Generator():
    def __init__(self, config, debug = False, shuffle=True, size = None):
        print('Loading dataset...')
        assert config['use_atlas'] == False
        dataset_root = config['dataset_root']
        val_split = config['val_split']
        self.debug = debug
        self.IDx = Task1Generator.get_idx(dataset_root, self.debug)
        
        if shuffle:
            np.random.shuffle(self.IDx)
        
        if size:
            self.new_IDx = self.IDx[0:size]
            train_size = int(len(self.new_IDx))
            train_idx = self.new_IDx[0:train_size]
            val_idx = self.IDx[train_size:int(train_size * (1+val_split) + 1)]      
        else:
            train_size = int(len(self.IDx) * (1-val_split))            
            train_idx = self.IDx[0:train_size]
            val_idx = self.IDx[train_size:]        

        train_idx_set = set(tuple(i) for i in train_idx)
        val_idx_set = set(tuple(i) for i in val_idx)
        IDx_set = set(tuple(i) for i in self.IDx)
        assert len(train_idx_set & val_idx_set) == 0
        assert len(train_idx_set & IDx_set) == len(train_idx)
        assert len(val_idx_set & IDx_set) == len(val_idx)
        print('Training set: {0}, Validation set: {1}'.format(len(train_idx), len(val_idx)))
        depth = config['depth']
        height = config['height']
        width = config['width']
        batch_size = config['batch_size']
        lowest = config['lowest']
        last = config['last']
        use_atlas = False
        
        self.train_generator = CustomDataGenerator(train_idx, depth, height, width, batch_size, lowest, last, use_atlas)
        self.val_generator = CustomDataGenerator(val_idx, depth, height, width, batch_size, lowest, last, use_atlas)
        

    def get_idx(dataset_root, debug = False):        
        allCases = glob.glob(dataset_root + '/EASY-RESECT/NIFTI/*', recursive= True)
        allCases.sort()
        IDx = []
        for case in allCases:
            number = re.findall('\d+', case)[-1]
            if number not in valid_items and debug:
                continue
            f_img = glob.glob(case + '/*FLAIR-resize.nii', recursive=True)
            t_img = glob.glob(case + '/*T1-resize.nii', recursive=True)
            assert len(f_img) == 1
            assert len(t_img) == 1
            IDx.append([f_img[0], t_img[0], None, None]) 
            IDx.append([t_img[0], f_img[0], None, None]) 
        return IDx