import argparse
import tensorflow as tf
from datetime import datetime
import tensorflow.keras.backend as K
import os

from src.create_model import create_model
from src.DataGenerators import Task1Generator, Task2Generator, Task3Generator, Task4Generator, MergeDataGenerator
from src.CustomTensorboard import Tensorboard_callback, LR_scheduler


task = [2,3,4]
use_atlas = True
config = {
    # Training setup
    'epochs':500,
    'lr':1e-4,
    'weights': None,
    'batch_size': 1,
    # Data 
    'depth': 64,
    'height': 64,
    'width': 64,
    'ds_size':300,
    'pretrain_size':50,
    'val_size':10,
    'use_atlas': use_atlas,
    'label_classes':14,
    'task': task,
    'dataset_root': [],
    # Architecture
    'lowest':4,
    'last':1,
    'pyramid_filters': [16,32,32,32],
    'use_affine': True,
    'use_def': True,
    'use_dense_net': True,    
    'use_context_net': False,
    'd': 2,
    'deform_filters':[64, 64, 32, 16, 8],
    # Loss
    'lambda': 5.0,
    'seg_loss': 'dice',
    'sim_loss': 'ncc',
    'smooth_loss': 'l2'
}

config['alphas'] = [1.0, 2.0, 4.0, 8.0, 16.0]
config['betas'] = [1.0, 0.5, 0.25, 0.125, 0.0625]

config['alphas'] = [i*0 for i in config['alphas']]
config['betas'] = [i*2 for i in config['betas']]

config['gamma'] = [1.0, 0.5, 0.25, 0.125, 0.0625]
config['gamma'] = [i*5 for i in config['gamma']]

def main(dataset_root):
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

    ds = MergeDataGenerator(generators, config, size=config['ds_size'], pretrain_size=config['pretrain_size'], val_size = config['val_size'], shuffle=True)
    
    if tf.config.experimental.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy() 
    config['batch_size'] *= strategy.num_replicas_in_sync
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "{0}_{1}_{2}".format('task{0}'.format(''.join(map(str,task))), config['sim_loss'], dt)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./Models/pretrained2_best',
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
    
    train_generator = ds.pretrain_generator
    num_batches = int(len(train_generator.idxs)/config['batch_size'])
    val_generator = None #ds.val_generator
    with strategy.scope():
        model, loss, loss_weights = create_model(config = config, name="Model")
        if config['weights'] is not None:
            model.load_weights(config['weights']).expect_partial()
        tensorboard = Tensorboard_callback(log_dir, config, train_generator, val_generator, model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss=loss, loss_weights=loss_weights)
        #print(len(ds.val_generator.idxs), config['batch_size'], int(len(ds.val_generator.idxs)/config['batch_size']))
        model.fit(train_generator.dataset,
                    callbacks = [tensorboard, model_checkpoint_callback],
                    epochs= config['epochs'], 
                    steps_per_epoch = num_batches)#,
                    #validation_data = val_generator.dataset,
                    #validation_steps = int(len(val_generator.idxs)/config['batch_size']))
        model.save_weights('./Models/pretrained2_model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", help="dataset root", type=str, required=True)
    parser.add_argument("-gpus", help="use gpus, example -1 (for cpu), 0 for gpu 0, 0,1,2 for gpu 0,1,2", type=str, default="1,2,3,4")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    dataset_root = args.ds
    main(dataset_root)
