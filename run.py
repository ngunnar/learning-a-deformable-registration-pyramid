from PWC_model import create_model
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
import losses
from DataGenerators import Task1Generator, Task2Generator, Task3Generator, Task4Generator, MergeDataGenerator
from datetime import datetime
import losses

from CustomTensorboard import Tensorboard_callback
from tensorflow.keras.utils import multi_gpu_model 
import tensorflow.keras.backend as K


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

task = [3]
use_atlas = True
config = {
    'depth': 64,
    'height': 64,
    'width': 64,
    'batch_size': 1,
    'ds_size':None,
    'use_affine': True,
    'use_def': True,
    'use_dense_net': True,    
    'use_context_net': False,
    'val_split':0.0,
    'epochs':5000,
    'lr':1e-4,
    'weights':None,
    'use_atlas': use_atlas,
    'atlas_wt': 1.0,
    'data_loss': 'ncc',
    'gamma':0.0004,
    'cost_search_range': 2,
    'lowest':4,
    'last':1,
    'task': task,
    'dataset_root': []
}

config['lambdas'] = [1.0, 0.25, 0.05, 0.0125, 0.002]
if config['data_loss'] == 'mse':
    config['reg_params'] = [50.0, 5.0, 2.5, 1.0, 0.5]
else:    
    config['reg_params'] = [1.0, 0.10, 0.05, 0.02, 0.01]

multi_gpu = False
debug = False


generators = []
if 1 in config['task']:
    generators.append([Task1Generator, '/data/Niklas/Learn2Reg/task_01'])
    config['dataset_root'].append('/data/Niklas/Learn2Reg/task_01')
if 2 in config['task']:
    generators.append([Task2Generator, '/data/Niklas/Learn2Reg/task_02'])
    config['dataset_root'].append('/data/Niklas/Learn2Reg/task_02')
if 3 in config['task']:
    generators.append([Task3Generator, '/data/Niklas/Learn2Reg/task_03'])
    config['dataset_root'].append('/data/Niklas/Learn2Reg/task_03')
if 4 in config['task']:
    generators.append([Task4Generator, '/data/Niklas/Learn2Reg/task_04'])
    config['dataset_root'].append('/data/Niklas/Learn2Reg/task_04')

ds = MergeDataGenerator(generators, config, debug, config['ds_size'],val=True)
     
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
pwc_model, loss, loss_weights = create_model(config = config, name="PWC_Net")
if config['weights'] is not None:
    pwc_model.load_weights(config['weights'])
pwc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss=loss, loss_weights=loss_weights)
#pwc_model.summary(line_length=150)
dt = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "{0}_{1}_{2}".format('task{0}'.format(''.join(map(str,task))), config['data_loss'], dt)
tensorboard = Tensorboard_callback(log_dir, config, ds, pwc_model)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./Saved_models/'+ dt +'/'+log_dir,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
pwc_model.fit_generator(generator = ds.train_generator,
                                     steps_per_epoch=None,
                                     epochs= config['epochs'],
                                     verbose=1,
                                     callbacks=[tensorboard, model_checkpoint_callback],
                                     validation_data = ds.val_generator if len(ds.val_generator) > 0 else None,
                                     validation_steps= len(ds.val_generator) if len(ds.val_generator) > 0 else None,
                                     validation_freq=1,
                                     class_weight=None,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False,
                                     shuffle=True,
                                     initial_epoch=0)

