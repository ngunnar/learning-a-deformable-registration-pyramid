import numpy as np
from .CustomDataGenerator import CustomDataGenerator

class MergeDataGenerator():
    
    def __init__(self, generators, config, debug, size = None, shuffle=True):
        print('Loading dataset...')
        val_split = config['val_split']
        self.debug = debug
        
        self.IDx = []
        train_idxs = []
        val_idxs = []
        for g, p in generators:
            idx = g.get_idx(p, self.debug)
            self.IDx.extend(idx)
            
            if shuffle:
                np.random.shuffle(self.IDx)
             
            if size:
                idx = idx[0:size]                       
            
            train_size = int(len(idx) * (1-val_split))            
            train_idx = idx[0:train_size]
            val_idx = idx[train_size:] 
            train_idx_set = set(tuple(i) for i in train_idx)
            val_idx_set = set(tuple(i) for i in val_idx)
            IDx_set = set(tuple(i) for i in idx)
            assert len(train_idx_set & val_idx_set) == 0
            assert len(train_idx_set & IDx_set) == len(train_idx)
            assert len(val_idx_set & IDx_set) == len(val_idx)
            train_idxs.extend(train_idx)
            val_idxs.extend(val_idx)
            print('Training set: {0}, Validation set: {1}'.format(len(train_idx), len(val_idx)))
        
        print('Training set: {0}, Validation set: {1}'.format(len(train_idxs), len(val_idxs)))
        
        depth = config['depth']
        height = config['height']
        width = config['width']
        batch_size = config['batch_size']
        lowest = config['lowest']
        last = config['last']
        use_atlas = config['use_atlas']

        self.train_generator = CustomDataGenerator(train_idxs, depth, height, width, batch_size, lowest, last, use_atlas)
        self.val_generator = CustomDataGenerator(val_idxs, depth, height, width, batch_size, lowest, last, use_atlas)