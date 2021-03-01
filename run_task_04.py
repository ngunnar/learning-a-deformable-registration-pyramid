#!/usr/bin/env python3
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
from model import Model
from DataGenerators import Task4Generator, MergeDataGenerator
import re
import time
import os

def main(fixed, moving, output):
    # load model
    model = Model(task_type=4)
    
    # load images
    fixed_img = nib.load(fixed).get_fdata()
    moving_img = nib.load(moving).get_fdata()
    
    # compute displacement field
    D, H, W = fixed_img.shape
    disp, t = model.predict(fixed_img, moving_img).astype('float16')
    # save displacement field
    np.savez_compressed(output, disp)

def run_all(dataset_root, save=True, t_type = 'test'):
    model = Model(task_type=4)
    gen = [Task4Generator, dataset_root]
    ds = MergeDataGenerator([gen], model.config, None, shuffle=False)
    assert t_type in ['test', 'val']
    if t_type == 'test':
        idxs = ds.test_generator.idxs
    else:
        idxs = ds.val_generator.idxs
    T = []
    for idx in idxs:
        fixed_path = idx[0][0]
        moving_path = idx[1][0]
        f_id = int(re.search(r'\d+', fixed_path[::-1]).group()[::-1])
        m_id = int(re.search(r'\d+', moving_path[::-1]).group()[::-1])
        print('Running task {0}, fixed {1}, moving {2}'.format(4, f_id, m_id))
        fixed_img = nib.load(fixed_path).get_fdata()
        moving_img = nib.load(moving_path).get_fdata()
        D, H, W = fixed_img.shape
        disp, t = model.predict(fixed_img, moving_img)
        T.append(t)
        if save:
            if not os.path.exists('./submission_{0}'.format(t_type)):
                os.makedirs('./submission_{0}'.format(t_type))
            if not os.path.exists('./submission_{0}/task_04'.format(t_type)):
                os.makedirs('./submission_{0}/task_04'.format(t_type))
            np.savez_compressed('./submission_{}/task_04/disp_{:04d}_{:04d}'.format(t_type, f_id, m_id), disp.astype('float16'))
    return T    
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-f', '--fixed', help="path to fixed image")
    parser.add_argument('-m', '--moving', help="path to moving image")
    parser.add_argument('-o', '--output', help="path to output displacement field")
    
    main(**vars(parser.parse_args()))