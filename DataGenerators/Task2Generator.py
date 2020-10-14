import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .utils import load_pairs

valid_items = ['001']

class Task2Generator():
    def get_task_number():
        return 2
    
    def get_test_idx(dataset_root):
        dataset_root = dataset_root + '/Test/task_02'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/training/scans/case_{:03d}_exp.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/training/scans/case_{:03d}_insp.nii.gz'.format(row['moving'])
            fixed_label = dataset_root+'/training/lungMasks/case_{:03d}_exp.nii.gz'.format(row['fixed'])
            moving_label = dataset_root+'/training/lungMasks/case_{:03d}_insp.nii.gz'.format(row['moving'])
            IDx.append([[fixed, fixed_label, [0, None]],
                        [moving, moving_label, [0, None]],
                        -1, None, Task2Generator.get_task_number()])
        return IDx
    
    def get_val_idx(dataset_root):
        dataset_root = dataset_root + '/task_02'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/training/scans/case_{:03d}_exp.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/training/scans/case_{:03d}_insp.nii.gz'.format(row['fixed'])
            fixed_label = dataset_root+'/training/lungMasks/case_{:03d}_exp.nii.gz'.format(row['fixed'])
            moving_label = dataset_root+'/training/lungMasks/case_{:03d}_insp.nii.gz'.format(row['fixed'])
            
            IDx.append([[fixed, fixed_label, [0, None]],
                        [moving, moving_label, [0, None]],
                        -1, None, Task2Generator.get_task_number()])
        return IDx
    
    def get_train_idx(dataset_root, use_atlas = True):
        dataset_root = dataset_root + '/task_02'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
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
            insp_img = [s for s in allInsp if number in s]
            assert len(insp_img) == 1
            if int(number) in pairs_task.values[:,0]:
                continue
            for l in allExp_Label:
                n = re.findall('\d+', l)[-1]
                if n != number:
                    continue
                exp_img = [s for s in allExp if n in s]
                assert len(exp_img) == 1
                for flip_ax in range(-1,3):
                    if use_atlas:
                        IDx.append([[insp_img[0], label, [0, None]], 
                                    [exp_img[0], l, [0, None]],
                                    flip_ax, None, Task2Generator.get_task_number()])
                        IDx.append([[exp_img[0], l, [0, None]], 
                                    [insp_img[0], label, [0, None]], 
                                    flip_ax, None, Task2Generator.get_task_number()])
                    else:
                        IDx.append([[insp_img[0], None, [0, None]], 
                                    [exp_img[0], None, [0, None]], 
                                    flip_ax, None])
                        IDx.append([[exp_img[0], None, [0, None]], 
                                    [insp_img[0], None, [0, None]],
                                    flip_ax, None, Task2Generator.get_task_number()])
                    '''        
                    for rot_ang in [-10, 0, 10]:
                        if use_atlas:
                            IDx.append([[insp_img[0], label, [0, None]], 
                                        [exp_img[0], l, [rot_ang, np.random.choice(3, 2, replace=False)]],
                                        flip_ax, None])
                            IDx.append([[exp_img[0], l, [0, None]], 
                                        [insp_img[0], label, [rot_ang, np.random.choice(3, 2, replace=False)]], 
                                        flip_ax, None])
                        else:
                            IDx.append([[insp_img[0], None, [0, None]], 
                                        [exp_img[0], None, [rot_ang, np.random.choice(3, 2, replace=False)]], 
                                        flip_ax, None])
                            IDx.append([[exp_img[0], None, [0, None]], 
                                        [insp_img[0], None, [rot_ang, np.random.choice(3, 2, replace=False)]],
                                        flip_ax, None])
                    '''                    
                                      
                
        return IDx
