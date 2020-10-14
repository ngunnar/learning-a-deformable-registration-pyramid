import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .utils import load_pairs

valid_items = ['1']

class Task1Generator():
    def get_task_number():
        return 1
    
    def get_test_idx(dataset_root):
        dataset_root = dataset_root + '/Test/task_01'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed_FLAIR = dataset_root+'/NIFTI/Case{0}/Case{0}-FLAIR-resize.nii'.format(row['fixed'])
            #fixed_T1 = dataset_root+'/NIFTI/Case{0}/Case{0}-T1-resize.nii'.format(row['moving'])
            moving_US = dataset_root+'/NIFTI/Case{0}/Case{0}-US-before-resize.nii'.format(row['moving'])
            IDx.append([[fixed_FLAIR, None, [0,None]], [moving_US, None, [0,None]], -1, None, Task1Generator.get_task_number()])
            #IDx.append([[fixed_T1, None, [0,None]], [moving_US, None, [0, None]], -1, None, Task1Generator.get_task_number()])
        return IDx
    
    def get_val_idx(dataset_root):
        dataset_root = dataset_root + '/task_01'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed_FLAIR = dataset_root+'/NIFTI/Case{0}/Case{0}-FLAIR-resize.nii'.format(row['fixed'])
            #fixed_T1 = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-T1-resize.nii'.format(row['moving'])
            moving_US = dataset_root+'/NIFTI/Case{0}/Case{0}-US-before-resize.nii'.format(row['moving'])
            IDx.append([[fixed_FLAIR, None, [0,None]], [moving_US, None, [0,None]], -1, None, Task1Generator.get_task_number()])
            #IDx.append([[fixed_T1, None, [0,None]], [moving_US, None, [0, None]], -1, None])
        return IDx
    
    def get_train_idx(dataset_root, use_atlas = True):
        assert use_atlas == False
        dataset_root = dataset_root + '/task_01'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        allMoving = [f for f in glob.glob(dataset_root + '/NIFTI/*/*-US-before-resize.nii', recursive= True)]
        allMoving.sort()
        allFixed = [f for f in glob.glob(dataset_root + '/NIFTI/*/*-FLAIR-resize.nii', recursive= True)]
        allFixed.sort()
        IDx = []
        for fixed in allFixed:
            number = re.findall('\d+', fixed)[-1]
            if int(number) in pairs_task.values[:,0]:
                continue
            moving = [s for s in allMoving if 'Case{0}-'.format(number) in s]
            assert len(moving) == 1, moving
            moving = moving[0]
            for flip_ax in range(-1,3):
                for rot_ang in [-10, 0, 10]:
                    IDx.append([[fixed, None, [0, None]], 
                                    [moving, None, [rot_ang, np.random.choice(3, 2, replace=False)]],
                                    flip_ax, None, Task1Generator.get_task_number()])
        return IDx    
