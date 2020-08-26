import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator

valid_items = ['001']

class Task2Generator():
    def __init__(self, config, debug = False, shuffle=True, size = None):
        print('Loading dataset...')
        dataset_root = config['dataset_root']
        val_split = config['val_split']
        self.debug = debug
        self.IDx = Task2Generator.get_idx(dataset_root, self.debug)
        
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
        use_atlas = config['use_atlas']
        
        self.train_generator = CustomDataGenerator(train_idx, depth, height, width, batch_size, lowest, last, use_atlas)
        self.val_generator = CustomDataGenerator(val_idx, depth, height, width, batch_size, lowest, last, use_atlas)
        

    def get_idx(dataset_root, debug = False):
        allInsp = [f for f in glob.glob(dataset_root + '/training/scans/case_*insp.nii.gz', recursive= True)]
        allInsp.sort()
        allExp = [f for f in glob.glob(dataset_root + '/training/scans/case_*exp.nii.gz', recursive= True)]
        allExp.sort()
        
        allInsp_Label = [f for f in glob.glob(dataset_root + '/training/lungMasks/case_*insp.nii.gz', recursive= True)]
        allInsp_Label.sort()
        allExp_Label = [f for f in glob.glob(dataset_root + '/training/lungMasks/case_*exp.nii.gz', recursive= True)]
        allInsp_Label.sort()
 
        IDx = []
        for label in allInsp_Label:
            number = re.findall('\d+', label)[-1]
            if number not in valid_items and debug:
                continue
            insp_img = [s for s in allInsp if number in s]
            assert len(insp_img) == 1
            for l in allExp_Label:
                n = re.findall('\d+', l)[-1]
                if n != number:
                    continue
                if n not in valid_items and debug:
                    continue
                exp_img = [s for s in allExp if n in s]
                assert len(exp_img) == 1
                IDx.append([insp_img[0], exp_img[0], label, l])
                IDx.append([exp_img[0], insp_img[0], l, label])
        return IDx