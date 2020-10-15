import argparse
import tensorflow as tf
from datetime import datetime
import tensorflow.keras.backend as K
import os

from src.create_model import create_model
from src.DataGenerators import Task1Generator, Task2Generator, Task3Generator, Task4Generator, MergeDataGenerator
from src.CustomTensorboard import Tensorboard_callback, LR_scheduler

task = [2]
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
    'val_split':0,
    'epochs':100,
    'lr':1e-5,
    'weights': './Models/pretrained_model',
    'use_atlas': use_atlas,
    'atlas_wt': 0,
    'seg_loss': 'dice',
    'data_loss': 'ncc',
    'gamma':0.0,
    'cost_search_range': 2,
    'lowest':4,
    'last':1,
    'task': task,
    'dataset_root': []
}

config['alphas'] = [1.0, 0.25, 0.05, 0.0125, 0.002]
config['betas'] = [1.0, 0.25, 0.05, 0.0125, 0.002]

config['alphas'] = [i for i in config['alphas']]
config['betas'] = [i*3 for i in config['betas']]
if config['data_loss'] == 'mse':
    config['reg_params'] = [50.0, 5.0, 2.5, 1.0, 0.5]
else:    
    config['reg_params'] = [1.0, 0.10, 0.05, 0.02, 0.01]
    config['reg_params'] = [i for i in config['reg_params']]

def main(dataset_root)
    generators = []
    if 1 in config['task']:
        generators.append([Task1Generator, dataset_root])
        config['dataset_root'].append(dataset_root+'/task_01')
    if 2 in config['task']:
        generators.append([Task2Generator, dataset_root])
        config['dataset_root'].append(dataset_root+'/task_02')
    if 3 in config['task']:
        generators.append([Task3Generator, dataset_root])
        config['dataset_root'].append(dataset_root + '/task_03')
    if 4 in config['task']:
        generators.append([Task4Generator, dataset_root])
        config['dataset_root'].append(dataset_root + '/task_04')

    ds = MergeDataGenerator(generators, config, config['ds_size'], shuffle=True)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    config['batch_size'] *= mirrored_strategy.num_replicas_in_sync
    num_batches = int(len(ds.train_generator.idxs)/config['batch_size'])
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "{0}_{1}_{2}".format('task{0}'.format(''.join(map(str,task))), config['data_loss'], dt)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./Models/task2_best',
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    with mirrored_strategy.scope():
        model, loss, loss_weights = create_model(config = config, name="Model")
        if config['weights'] is not None:
            model.load_weights(config['weights']).expect_partial()
        tensorboard = Tensorboard_callback(log_dir, config, ds, model)
        lr_scheduler = LR_scheduler(config['lr'], config['epochs'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss=loss, loss_weights=loss_weights)
        model.fit(ds.train_generator.dataset,
                    callbacks = [tensorboard, lr_scheduler, model_checkpoint_callback],
                    epochs= config['epochs'], 
                    steps_per_epoch = num_batches,
                    validation_data = ds.val_generator.dataset,
                    validation_steps = config['batch_size'])
        model.save_weights('./Models/task2_model_new')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", help="dataset root", type=str, required=True)
    parser.add_argument("-gpus", help="use gpus, example -1 (for cpu), 0 for gpu 0, 0,1,2 for gpu 0,1,2", type=str, default="1,2,3,4")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    dataset_root = args.ds
    main(dataset_root)