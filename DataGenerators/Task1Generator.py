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
            fixed_FLAIR = dataset_root+'/CuRIOUS2020-TestData/NIFTI/TestCase{0}-FLAIR-resize.nii'.format(row['fixed'])
            fixed_T1 = dataset_root+'/CuRIOUS2020-TestData/NIFTI/TestCase{0}-T1w-resize.nii'.format(row['moving'])
            moving_US = dataset_root+'/CuRIOUS2020-TestData/NIFTI/TestCase{0}-US-before-resize.nii'.format(row['moving'])
            IDx.append([[fixed_FLAIR, None, [0,None]], [moving_US, None, [0,None]], -1, None, Task1Generator.get_task_number()])
            IDx.append([[fixed_T1, None, [0,None]], [moving_US, None, [0, None]], -1, None, Task1Generator.get_task_number()])
        return IDx
    
    def get_val_idx(dataset_root):
        dataset_root = dataset_root + '/task_01'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed_FLAIR = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-FLAIR-resize.nii'.format(row['fixed'])
            #fixed_T1 = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-T1-resize.nii'.format(row['moving'])
            moving_US = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-US-before-resize.nii'.format(row['moving'])
            IDx.append([[fixed_FLAIR, None, [0,None]], [moving_US, None, [0,None]], -1, None, Task1Generator.get_task_number()])
            #IDx.append([[fixed_T1, None, [0,None]], [moving_US, None, [0, None]], -1, None])
        return IDx
    
    def get_train_idx(dataset_root, use_atlas = True):
        assert use_atlas == False
        dataset_root = dataset_root + '/task_01'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed_FLAIR = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-FLAIR-resize.nii'.format(row['fixed'])
            fixed_T1 = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-T1-resize.nii'.format(row['moving'])
            moving_US = dataset_root+'/EASY-RESECT/NIFTI/Case{0}/Case{0}-US-before-resize.nii'.format(row['moving'])
            IDx.append([[fixed_FLAIR, None, [0,None]], [moving_US, None, [0,None]], -1, None, Task1Generator.get_task_number()])
            IDx.append([[fixed_T1, None, [0,None]], [moving_US, None, [0, None]], -1, None, Task1Generator.get_task_number()])
        return IDx    
