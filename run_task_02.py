#!/usr/bin/env python3
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
from model import Model
from DataGenerators import Task2Generator, MergeDataGenerator
import re
import time
import os

def main(exh, inh, exh_mask, inh_mask, output):
    # load model
    model = Model(task_type=2)
    
    # load images
    exh_img = nib.load(exh).get_fdata()
    inh_img = nib.load(inh).get_fdata()
    exh_mask_img = nib.load(exh_mask).get_fdata()
    inh_mask_img = nib.load(inh_mask).get_fdata()

    # compute displacement field
    D, H, W = exh_img.shape
    disp, t = model.predict(exh_img, inh_img)
    disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
    disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
    disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
    disp = np.array((disp_x, disp_y, disp_z))
    
    # save displacement field
    np.savez_compressed(output, disp)

    
def run_all(dataset_root, save = True, t_type = 'test'):
    model = Model(task_type=2)
    gen = [Task2Generator, dataset_root]
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
        print('Running task {0}, fixed {1}, moving {2}'.format(2, f_id, m_id))
        exh_img = nib.load(fixed_path).get_fdata()
        inh_img = nib.load(moving_path).get_fdata()
        D, H, W = exh_img.shape
        disp, t = model.predict(exh_img, inh_img)
        T.append(t)
        disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
        disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
        disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
        disp = np.array((disp_x, disp_y, disp_z))
        if save:
            if not os.path.exists('./submission_{0}'.format(t_type)):
                os.makedirs('./submission_{0}'.format(t_type))
            if not os.path.exists('./submission_{0}/task_02'.format(t_type)):
                os.makedirs('./submission_{0}/task_02'.format(t_type))
            np.savez_compressed('./submission_{}/task_02/disp_{:04d}_{:04d}'.format(t_type, f_id, m_id), disp)
    return T
        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-e', '--exh', help="path to fixed image (exh)")
    parser.add_argument('-i', '--inh', help="path to moving image (inh)")
    parser.add_argument('-em', '--exh_mask', help="path to fixed mask (exh)")
    parser.add_argument('-im', '--inh_mask', help="path to moving mask (inh)")
    parser.add_argument('-o', '--output', help="path to output displacement field")
    
    main(**vars(parser.parse_args()))