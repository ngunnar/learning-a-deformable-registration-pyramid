import numpy as np
from .CustomDataGenerator import CustomDataGenerator
from .TensorflowDatasetLoader import TensorflowDatasetLoader

class MergeDataGenerator():    
    def __init__(self, generators, config, size = None, shuffle=True, t_type = 'train'):
        print('Loading dataset...')
        assert t_type in ['test', 'train', 'val'], t_type
        val_split = config['val_split']
        use_atlas = config['use_atlas']
        
        self.IDx = []
        train_idxs = []
        val_idxs = []
        for g, p in generators:
            if t_type == 'test':
                idx = g.get_test_idx(p)
            elif t_type == 'train':
                idx = g.get_train_idx(p, use_atlas)
            else:
                idx = g.get_val_idx(p)
                
            if size:
                idx = idx[0:size] 
            
            if shuffle:
                np.random.shuffle(idx)
            
            self.IDx.extend(idx)

            train_size = int(len(idx) * (1-val_split))            
            train_idx = idx[0:train_size]
            val_idx = idx[train_size:] 
            '''
            train_idx_set = set(tuple(i) for i in train_idx)
            val_idx_set = set(tuple(i) for i in val_idx)
            IDx_set = set(tuple(i) for i in idx)
            assert len(train_idx_set & val_idx_set) == 0
            assert len(train_idx_set & IDx_set) == len(train_idx), len(train_idx)
            assert len(val_idx_set & IDx_set) == len(val_idx)
            '''
            train_idxs.extend(train_idx)
            val_idxs.extend(val_idx)
            
            print('\tTraining set: {0}, Validation set: {1}'.format(len(train_idx), len(val_idx)))
        
        print('Training set: {0}, Validation set: {1}'.format(len(train_idxs), len(val_idxs)))
        
        if shuffle:
            np.random.shuffle(train_idxs)
            np.random.shuffle(val_idx)
        
        depth = config['depth']
        height = config['height']
        width = config['width']
        batch_size = config['batch_size']
        lowest = config['lowest']
        last = config['last']
        
        self.train_generator = TensorflowDatasetLoader(train_idxs, depth, height, width, batch_size, lowest, last, use_atlas)
        self.val_generator = TensorflowDatasetLoader(val_idxs, depth, height, width, batch_size, lowest, last, use_atlas)