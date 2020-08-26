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

def comp(model, config):
    
    def w_loss(loss):
        def l(_, yp):
            i = yp[..., :yp.shape[-1]//2]
            w = yp[..., yp.shape[-1]//2:]
            return loss(i, w)
        return l
        
    d_l = config['data_loss']
    assert d_l in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    
    if d_l in ['ncc', 'cc']:
        d_l = losses.NCC().loss
    else:
        d_l = tf.keras.losses.MeanSquaredError()
        
    loss = []
    loss_weights = []
    for i in range(config['lowest'] - config['last'] + 1):
        l = config['lowest'] - i
        print(l, config['reg_param']/(2**l))
        loss.append(losses.Grad('l2').loss)
        loss_weights.append(config['lambda']/(2**l))
        loss.append(w_loss(d_l))
        loss_weights.append(config['reg_param']/(2**l))
            

    loss.append(losses.Grad('l2').loss)
    loss_weights.append(config['lambda'])

    loss.append(d_l)
    loss_weights.append(config['reg_param'])
    if config['use_atlas']:
        loss.append(losses.Dice().loss)#config['seg_no']).loss)
        loss_weights.append(config['atlas_wt'])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss=loss, loss_weights=loss_weights)

task = [2,3,4]
config = {
    'depth': 64,
    'height': 64,
    'width': 64,
    'batch_size': 1,
    'ds_size':100,
    'use_dense_net': True,
    'use_context_net': False,
    'task': 3,
    'val_split':0.03,
    'epochs':5000,
    'lr':1e-4,
    'weights':None,
    'use_atlas': True, #task != 1,
    'atlas_wt': 1.0,
    'data_loss': 'ncc',
    'gamma':0.00, #0.0004,
    'cost_search_range': 2,
    'lowest':4,
    'last':1,
    'task': task,
}
    
if config['data_loss'] == 'mse':
    config['lambda'] = 0.1
    config['reg_param'] = 100.0
else:
    config['lambda'] = 1.0
    config['reg_param'] = 10.0
    
multi_gpu = False
debug = False

generators = []
if 1 in config['task']:
    generators.append([Task1Generator, '/data/Niklas/Learn2Reg/task_01'])
if 2 in config['task']:
    generators.append([Task2Generator, '/data/Niklas/Learn2Reg/task_02'])
if 3 in config['task']:
    generators.append([Task3Generator, '/data/Niklas/Learn2Reg/task_03'])
if 4 in config['task']:
    generators.append([Task4Generator, '/data/Niklas/Learn2Reg/task_04'])

ds = MergeDataGenerator(generators, config, debug, config['ds_size'])
'''
assert config['task'] in [1, 2, 3, 4], 'task should be 1,2, 3 or 4' % config['task']
if config['task'] == 1:
    ds = Task1Generator(config, debug=debug)
elif config['task'] == 2:
    ds = Task2Generator(config, debug=debug)
    config['seg_no'] = 2
elif config['task'] == 3:
    ds = Task3Generator(config, debug=debug)
    config['seg_no'] = 14
elif config['task'] == 4:
        ds = Task4Generator(config, debug=debug)
        config['seg_no'] = 3
'''
pwc_model = create_model(config = config, name="PWC_Net")
pwc_model.summary(line_length=150)
comp(pwc_model, config)
log_dir = "{0}_{1}_{2}_{3}".format('task{0}'.format(task), datetime.now().strftime("%Y%m%d-%H%M%S"), 'short', config['use_dense_net'])
tensorboard = Tensorboard_callback(log_dir, config, ds, pwc_model)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./Saved models/'+log_dir,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

pwc_model.fit_generator(
    generator = ds.train_generator,
    steps_per_epoch=None,
    epochs= config['epochs'],
    verbose=1,
    callbacks=[tensorboard, model_checkpoint_callback],
    validation_data= ds.val_generator,
    validation_steps= len(ds.val_generator),
    validation_freq=1,
    class_weight=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    shuffle=True,
    initial_epoch=0)
