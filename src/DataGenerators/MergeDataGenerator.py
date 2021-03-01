import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .TensorflowDatasetLoader import TensorflowDatasetLoader


def add(idx, fixed, moving, fixed_label, moving_label, flip_ax, rot, rot_ax, crop_img, use_altas, use_both, task_no):
    if not use_altas:
        fixed_label = None
        moving_label = None
    
    idx.append([[fixed, fixed_label, [rot, rot_ax]],
                    [moving, moving_label, [rot, rot_ax]],
                    flip_ax, crop_img, task_no])
    if use_both:
        idx.append([[moving, moving_label, [rot, rot_ax]],
                    [fixed, fixed_label, [rot, rot_ax]],
                    flip_ax, crop_img, task_no])
        

class MergeDataGenerator():    
    def __init__(self, generators, config, size = None, pretrain_size=None, val_size = None, test_size = None, shuffle=True, init=False):
        print('Loading dataset...')
        #assert t_type in ['test', 'train', 'val'], t_type
        #val_split = config['val_split']
        use_atlas = config['use_atlas']
        
        pretrain_idxs = []
        train_idxs = []
        val_idxs = []
        test_idxs = []
        for g, p in generators:
            test_idx = g.get_test_idx(p)
            pretrain_idx = g.get_train_idx(p, use_atlas, True, True) #flip, use_both only for task2
            train_idx = g.get_train_idx(p, use_atlas, True, True)
            val_idx = g.get_val_idx(p)
            
            if pretrain_size:
                pretrain_idx = pretrain_idx[0:pretrain_size]            
            if size:
                train_idx = train_idx[0:size] 
            if val_size:
                val_idx = val_idx[0:val_size]
            if test_size:
                test_idx = test_idx[0:test_size]
            #if shuffle:
            #    np.random.shuffle(idx)
            
            #self.IDx.extend(idx)

            #train_size = int(len(idx) * (1-val_split))            
            #train_idx = idx[0:train_size]
            #val_idx = idx[train_size:] 
            '''
            train_idx_set = set(tuple(i) for i in train_idx)
            val_idx_set = set(tuple(i) for i in val_idx)
            IDx_set = set(tuple(i) for i in idx)
            assert len(train_idx_set & val_idx_set) == 0
            assert len(train_idx_set & IDx_set) == len(train_idx), len(train_idx)
            assert len(val_idx_set & IDx_set) == len(val_idx)
            '''
            pretrain_idxs.extend(pretrain_idx)
            train_idxs.extend(train_idx)
            val_idxs.extend(val_idx)
            test_idxs.extend(test_idx)
            
            print('\tPretrain set: {0}, Training set: {1}, Validation set: {2}, Test set: {3}'.format(len(pretrain_idx), len(train_idx), len(val_idx), len(test_idx)))
         
        print('Pretrain set: {0}, Training set: {1}, Validation set: {2}, Test set: {3}'.format(len(pretrain_idxs), len(train_idxs), len(val_idxs), len(test_idxs)))
        if shuffle:
            np.random.shuffle(pretrain_idxs)
            np.random.shuffle(train_idxs)
            #np.random.shuffle(val_idx)
            #np.random.shuffle(test_idx)
        
        depth = config['depth']
        height = config['height']
        width = config['width']
        batch_size = config['batch_size']
        lowest = config['lowest']
        last = config['last']
        
        self.pretrain_generator = TensorflowDatasetLoader(pretrain_idxs, config)
        self.train_generator = TensorflowDatasetLoader(train_idxs, config)
        self.val_generator = TensorflowDatasetLoader(val_idxs, config)
        self.test_generator = TensorflowDatasetLoader(test_idxs, config)