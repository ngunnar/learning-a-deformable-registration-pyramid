#!/usr/bin/python

import glob
import numpy as np
import os
from scipy.ndimage.interpolation import zoom as zoom
import shutil
import sys

def main():
    INPUT_FOLDER = sys.argv[1]
    OUTPUT_FOLDER = INPUT_FOLDER + '_compressed'

    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_01'))
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_02'))
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_03'))
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_04'))
        
    for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_01', '*.npy')):
        print('compressing {}...'.format(file))
        disp = np.load(file) #expects shape 3x256x256x288
        disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
        disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
        disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
        disp = np.array((disp_x, disp_y, disp_z))
        np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
        
    for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_02', '*.npy')):
        print('compressing {}...'.format(file))
        disp = np.load(file) #expects shape 3x192x192x208
        disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
        disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
        disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
        disp = np.array((disp_x, disp_y, disp_z))
        np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
        
    for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_03', '*.npy')):
        print('compressing {}...'.format(file))
        disp = np.load(file) #expects shape 3x192x160x256
        disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
        disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
        disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
        disp = np.array((disp_x, disp_y, disp_z))
        np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
        
    for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_04', '*.npy')):
        print('compressing {}...'.format(file))
        disp = np.load(file) #expects shape 3x64x64x64
        disp = disp.astype('float16')
        np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
        
    shutil.make_archive('submission', 'zip', OUTPUT_FOLDER)
    
    print('...finished creating submission.')

if __name__ == "__main__":
    main()