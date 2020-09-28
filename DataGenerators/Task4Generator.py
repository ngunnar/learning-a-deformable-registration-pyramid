import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .utils import load_pairs

class Task4Generator():
    def get_task_number():
        return 4
    
    def get_test_idx(dataset_root):
        dataset_root = dataset_root + '/Test/task_04'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/Testing/img/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/Testing/img/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            IDx.append([[fixed, None, [0, None]],
                            [moving, None, [0, None]],
                            -1, None, Task4Generator.get_task_number()])          
        return IDx  
   
    def get_val_idx(dataset_root):
        dataset_root = dataset_root + '/task_04'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/L2R_task4_files/Training/img/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/L2R_task4_files/Training/img/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            fixed_label = dataset_root+'/L2R_task4_files/Training/label/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving_label = dataset_root+'/L2R_task4_files/Training/label/hippocampus_{:03d}.nii.gz'.format(row['moving'])            
            IDx.append([[fixed, fixed_label, [0, None]],
                            [moving, moving_label, [0, None]],
                            -1, None, Task4Generator.get_task_number()])          
        return IDx
    
    def get_train_idx(dataset_root, use_atlas = True):
        dataset_root = dataset_root + '/task_04'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/L2R_task4_files/Training/img/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/L2R_task4_files/Training/img/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            fixed_label = dataset_root+'/L2R_task4_files/Training/label/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving_label = dataset_root+'/L2R_task4_files/Training/label/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            for flip_ax in range(-1,3):
                if use_atlas:
                    IDx.append([[fixed, fixed_label, [0, None]], 
                           [moving, moving_label, [0, None]],
                            -1, 6, Task4Generator.get_task_number()])
                    IDx.append([[moving, moving_label, [0, None]], 
                           [fixed, fixed_label, [0, None]],
                            -1, 6, Task4Generator.get_task_number()])                    
                else:
                    IDx.append([[fixed, None, [0, None]],
                            [moving, None, [0, None]],
                            -1, 6, Task4Generator.get_task_number()])
                    IDx.append([[moving, None, [0, None]],
                            [fixed, None, [0, None]],
                            -1, 6, Task4Generator.get_task_number()])
                '''    
                for rot_ang in [-10, 0, 10]:
                    if use_atlas:
                        IDx.append([[fixed, fixed_label, [0, None]], 
                                    [moving, moving_label, [rot_ang, np.random.choice(3, 2, replace=False)]], 
                                    flip_ax, 6])
                        IDx.append([[moving, moving_label, [0, None]],
                                    [fixed, fixed_label, [rot_ang, np.random.choice(3, 2, replace=False)]],
                                    flip_ax, 6])
                    else:
                        IDx.append([[fixed, None, [0, None]], 
                                    [moving, None, [rot_ang, np.random.choice(3, 2, replace=False)]],
                                    flip_ax, 6])
                        IDx.append([[moving, None, [0, None]],
                                    [fixed, None, [rot_ang, np.random.choice(3, 2, replace=False)]],
                                    flip_ax, 6])
                  '''
        return IDx