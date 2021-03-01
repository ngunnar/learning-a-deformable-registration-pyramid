#!/usr/bin/env python3
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
from model import Model
from DataGenerators import Task1Generator, MergeDataGenerator
import re
import time
import os

def main(flair, t1, us, output):
    # load model
    model, t = Model(task_type=1)
    # load images
    flair_img = nib.load(flair).get_fdata()
    t1_img = nib.load(t1).get_fdata()
    us_img = nib.load(us).get_fdata()
    
    # compute displacement field
    D, H, W = flair_img.shape
    disp = model.predict(flair_img, us_img)
    disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
    disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
    disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
    disp = np.array((disp_x, disp_y, disp_z))
    
    # save displacement field
    np.savez_compressed(output, disp)

    
def run_all(dataset_root, save = True, t_type = 'test'):
    model = Model(task_type=4)
    gen = [Task1Generator, dataset_root]
    ds = MergeDataGenerator([gen], model.config, None, shuffle=False)
    T = []
    assert t_type in ['test', 'val']
    if t_type == 'test':
        idxs = ds.test_generator.idxs
    else:
        idxs = ds.val_generator.idxs
    for idx in idxs:
        fixed_path = idx[0][0]
        moving_path = idx[1][0]
        f_id = int(re.search(r'\d+', fixed_path[::-1]).group()[::-1])
        m_id = int(re.search(r'\d+', moving_path[::-1]).group()[::-1])
        print('Running task {0}, fixed {1}, moving {2}'.format(1, f_id, m_id))
        fixed_img = nib.load(fixed_path).get_fdata()
        moving_img = nib.load(moving_path).get_fdata()
        t0 = time.time()
        D, H, W = fixed_img.shape
        disp = np.zeros((3, D, H, W))
        T.append(time.time() - t0)
        disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
        disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
        disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
        disp = np.array((disp_x, disp_y, disp_z))
        if save:
            if not os.path.exists('./submission_{0}'.format(t_type)):
                os.makedirs('./submission_{0}'.format(t_type))
            if not os.path.exists('./submission_{0}/task_01'.format(t_type)):
                os.makedirs('./submission_{0}/task_01'.format(t_type))
            np.savez_compressed('./submission_{}/task_01/disp_{:04d}_{:04d}'.format(t_type, f_id, m_id), disp)
    return T 
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-f', '--flair', help="path to fixed image (FLAIR)")
    parser.add_argument('-t', '--t1', help="path to fixed image (T1)")
    parser.add_argument('-u', '--us', help="path to moving image (US)")
    parser.add_argument('-o', '--output', help="path to output displacement field")
    
    main(**vars(parser.parse_args()))