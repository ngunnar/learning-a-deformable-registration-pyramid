import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .utils import load_pairs

valid_items = ['311', '394']

class Task4Generator():
    def __init__(self, config, debug = False, shuffle=True, size = None, val = False):
        print('Loading dataset...')
        dataset_root = config['dataset_root']
        val_split = config['val_split']
        self.use_atlas = config['use_atlas']
        self.debug = debug
        if val:
            self.IDx = Task4Generator.get_val_idx(dataset_root, self.use_atlas)
        else:
            self.IDx = Task4Generator.get_idx(dataset_root, self.debug, self.use_atlas)
        
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
        
        self.train_generator = CustomDataGenerator(train_idx, depth, height, width, batch_size, lowest, last)
        self.val_generator = CustomDataGenerator(val_idx, depth, height, width, batch_size, lowest, last)
        
    def get_val_idx(dataset_root, use_atlas = True):
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/L2R_task4_files/Training/img/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/L2R_task4_files/Training/img/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            fixed_label = dataset_root+'/L2R_task4_files/Training/label/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving_label = dataset_root+'/L2R_task4_files/Training/label/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            if use_atlas:
                IDx.append([fixed, moving, fixed_label, moving_label])
                IDx.append([moving, fixed, moving_label, fixed_label])
            else:
                IDx.append([fixed, moving, None, None])
                IDx.append([moving, fixed, None, None])
        return IDx            
        
    def get_idx(dataset_root, debug = False, use_atlas = True):        
        allImages = [f for f in glob.glob(dataset_root + '/L2R_task4_files/Training/img/*.nii.gz', recursive= True)]
        allImages.sort()
        allLabel = [f for f in glob.glob(dataset_root + '/L2R_task4_files/Training/label/*.nii.gz', recursive= True)]
        allLabel.sort()
 
        IDx = []
        for label in allLabel:
            number = re.findall('\d+', label)[-1]
            if number not in valid_items and debug:
                continue
            fixed = [s for s in allImages if number in s]
            assert len(fixed) == 1
            for l in allLabel:
                n = re.findall('\d+', l)[-1]
                if n == number:
                    continue
                if n not in valid_items and debug:
                    continue
                moving = [s for s in allImages if n in s]
                assert len(moving) == 1
                if use_atlas:
                    IDx.append([fixed[0], moving[0], label, l])
                else:
                    IDx.append([fixed[0], moving[0], None, None])
        return IDx