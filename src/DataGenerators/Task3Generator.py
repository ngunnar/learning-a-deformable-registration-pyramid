import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .utils import load_pairs

valid_items = ['0001', '0004']

class Task3Generator():
    def get_task_number():
        return 3   
    def get_test_idx(dataset_root):
        dataset_root = dataset_root + '/Test/task_03'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/Training/img/img{:04d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/Training/img/img{:04d}.nii.gz'.format(row['moving'])
            IDx.append([[fixed, None, [0, None]],
                            [moving, None, [0, None]],
                            -1, None, Task3Generator.get_task_number()])          
        return IDx
    
    def get_val_idx(dataset_root):
        dataset_root = dataset_root + '/task_03'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/Training/img/img{:04d}.nii.gz'.format(row['fixed'])
            fixed_label = dataset_root+'/Training/label/label{:04d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/Training/img/img{:04d}.nii.gz'.format(row['moving'])
            moving_label = dataset_root+'/Training/label/label{:04d}.nii.gz'.format(row['moving'])
            IDx.append([[fixed, fixed_label, [0, None]],
                            [moving, moving_label, [0, None]],
                            -1, None, Task3Generator.get_task_number()])          
        return IDx
    
    def get_train_idx(dataset_root, use_atlas = True):
        dataset_root = dataset_root + '/task_03'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        allImages = [f for f in glob.glob(dataset_root + '/Training/img/*.nii.gz', recursive= True)]
        allImages.sort()
        allLabel = [f for f in glob.glob(dataset_root + '/Training/label/*.nii.gz', recursive= True)]
        allLabel.sort()
 
        IDx = []
        for label in allLabel:
            number = re.findall('\d+', label)[-1]
            fixed = [s for s in allImages if number in s]
            assert len(fixed) == 1
            f_val = int(number) in pairs_task.values[:,0]
            for l in allLabel:
                n = re.findall('\d+', l)[-1]
                if n == number:
                    continue
                moving = [s for s in allImages if n in s]
                assert len(moving) == 1
                m_val = int(number) in pairs_task.values[:,1]
                if f_val and m_val:
                    continue
                if use_atlas:
                    IDx.append([[fixed[0], label, [0, None]], 
                           [moving[0], l, [0, None]],
                            -1, None, Task3Generator.get_task_number()])
                else:
                    IDx.append([[fixed[0], None, [0, None]],
                            [moving[0], None, [0, None]],
                            -1, None, Task3Generator.get_task_number()])
        return IDx