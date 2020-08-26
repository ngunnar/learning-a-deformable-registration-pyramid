dicom_config = {
    'height': 512,
    'width': 512,
    'batch_size': 4,
    'ds_size': None,
    'use_dense_net': True,
    'use_context_net': True,
    'dataset':'DICOM',
    's_type':'short', # short, long or fine
    'dataset_root': '/data/Flow/MRLinac_flow',
    'val_split':0.03,
    'iterations':5000,
    'start_iteration':0,
    'lr':1e-4,
    'weights':'./Saved models/pwc_model_Chairs.h5',
    'loss_weights':[0.0, 0.005, 0.01, 0.02, 0.08, 0.32], #
    'n':1,
    'q':0.4,
    'epsilon':0.01,
    'gamma':0.0004
}

flyingthings3D_config = {
    'height': 256,
    'width': 256,
    'batch_size': 4,
    'ds_size': 5,
    'use_dense_net': True,
    'use_context_net': True,
    'dataset':'FlyingThings3D',
    's_type':'short', # short, long or fine
    'dataset_root': '/data/Flow/FlyingThings3D',
    'val_split':0.03,
    'iterations':6e5,
    'start_iteration':0,
    'lr':1e-4,
    'weights':None,
    'loss_weights':[0.0, 0.005, 0.01, 0.02, 0.08, 0.32],
    'n':1,
    'q':0.4,
    'epsilon':0.01,
    'gamma':0.0004
}

flyingChairs_config = {
    'height': 512,
    'width': 512,
    'batch_size': 4,
    'ds_size': 5,
    'use_dense_net': True,
    'use_context_net': True,
    'dataset':'FlyingChairs',
    's_type':'short', # short, long or fine
    'dataset_root': '/data/Flow/FlyingChairs_release',
    'val_split':0.03,
    'iterations':6e5,
    'start_iteration':0,
    'lr':1e-4,
    'weights':None,
    'loss_weights':[0.0, 0.005, 0.01, 0.02, 0.08, 0.32],
    'n':2,
    'q':1.0,
    'epsilon':0.0,
    'gamma':0.0004
}

mixed_config = {
    'height': 512,
    'width': 512,
    'batch_size': 4,
    'ds_size': None,
    'use_dense_net': True,
    'use_context_net': True,
    'dataset':'Mixed',
    's_type':'short', # short, long or fine
    'dataset_root': ['/data/Flow/FlyingThings3D', '/data/Flow/FlyingChairs_release'],
    'val_split':0.03,
    'iterations':6e5,
    'start_iteration':0,
    'lr':1e-4,
    'weights':None,
    'loss_weights':[0.0, 0.005, 0.01, 0.02, 0.08, 0.32],
    'n':2,
    'q':1.0,
    'epsilon':0.0,
    'gamma':0.0004
}