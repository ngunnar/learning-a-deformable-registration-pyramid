import re
import os
import glob
import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .utils import load_pairs
from .MergeDataGenerator import add

class Task4Generator():
    def get_task_number():
        return 4
    
    def get_test_idx(dataset_root):
        task_no = Task4Generator.get_task_number()
        dataset_root = dataset_root + '/Test/task_04'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/Training/img/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/Training/img/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            add(IDx, fixed, moving, None, None, -1, 0, None, None, True, False, task_no)         
        return IDx  
   
    def get_val_idx(dataset_root):
        task_no = Task4Generator.get_task_number()
        dataset_root = dataset_root + '/task_04'
        pairs_task = load_pairs(dataset_root+'/pairs_val.csv')
        IDx = []
        for _, row in pairs_task.iterrows():
            fixed = dataset_root+'/Training/img/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving = dataset_root+'/Training/img/hippocampus_{:03d}.nii.gz'.format(row['moving'])
            fixed_label = dataset_root+'/Training/label/hippocampus_{:03d}.nii.gz'.format(row['fixed'])
            moving_label = dataset_root+'/Training/label/hippocampus_{:03d}.nii.gz'.format(row['moving'])  
            add(IDx, fixed, moving, fixed_label, moving_label, -1, 0, None, None, True, False, task_no)       
        return IDx
    

    def get_train_idx(dataset_root, use_atlas = True, flip = True, use_both = True):
        task_no = Task4Generator.get_task_number()
        dataset_root = dataset_root + '/task_04'
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
                
                add(IDx, fixed[0], moving[0], label, l, -1, 0, None, 6, use_atlas, False, task_no)   
        return IDx
