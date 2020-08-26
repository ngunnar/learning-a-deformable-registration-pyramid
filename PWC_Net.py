import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, Callback, LambdaCallback
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

from DataGenerators.FlyingChairDatasetGenerator import FlyingChairDatasetGenerator
from DataGenerators.FlyingThings3dGenerator import FlyingThings3dGenerator
from DataGenerators.MixedGenerator import MixedGenerator
from DataGenerators.DicomGenerator import DicomGenerator

from losses import epe, loss, get_score

from Utils import tf_Utils
from PWC_model import create_model

import matplotlib.pyplot as plt
import os, math, io, time
import datetime
import numpy as np

class PWC_Net(object):
    def __init__(self, config, name):
        assert config['s_type'] in ['short', 'long', 'fine'], config['type']
        assert config['dataset'] in ['FlyingChairs', 'FlyingThings3D', 'Mixed', 'DICOM'], config['dataset']
        self.config = config
        #################### LOADING DATA #########################################
        print('Loading data ...')
        if self.config['dataset'] == 'FlyingChairs':
            self.ds = FlyingChairDatasetGenerator(config=config, size=self.config['ds_size'])
        elif self.config['dataset'] == 'FlyingThings3D':
            self.ds = FlyingThings3dGenerator(config=config, size=self.config['ds_size'])
        elif self.config['dataset'] == 'DICOM':
            self.ds = DicomGenerator(config=config, size=self.config['ds_size'])
        else:
            self.ds = MixedGenerator(config=config, size=self.config['ds_size'])
        print("\t Data loaded!")        
        
        #################### CREATING MODEL #########################################
        print("Creating model...")        
        self.model = create_model(self.config, name)
        print("\t Model created!")
        
        #################### LOAD WEIGHTS #########################################         
        if self.config['weights'] != None:
            print("Loading weights...")
            self.model.load_weights(self.config['weights'])
            print("Weights loaded!")        
        
        #################### COMPILING MODEL #########################################
        print("Compiling model...")
        self.model.compile(loss=loss(self.config),
                  loss_weights=self.config['loss_weights'], 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                  metrics={'est_flow':epe })        
        print("\t Model compiled!")       
    
    def run(self, callbacks):
        #iterations = self.model.optimizer.iterations.numpy()
        iteration_per_epoch = len(self.ds.train_generator)
        iterations = self.config['iterations']
        epochs = int(iterations // iteration_per_epoch)
        start_epoch = int(self.config['start_iteration'] // iteration_per_epoch)       
        print("Training model. Starting on epoch {0}".format(start_epoch))
        return self.model.fit_generator(generator = self.ds.train_generator,
                                 steps_per_epoch=None,
                                 epochs= epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=self.ds.val_generator,
                                 validation_steps= len(self.ds.val_generator),
                                 validation_freq=1,
                                 class_weight=None,
                                 max_queue_size=10,
                                 workers=1,
                                 use_multiprocessing=False,
                                 shuffle=True,
                                 initial_epoch=start_epoch)


def Tensorboard_callback(log_dir, config, ds, model):
    class CustomTensorBoard(TensorBoard):
        def __init__(self, **kwargs):  # add other arguments to __init__ if you need
            super().__init__(**kwargs)
        
        def on_epoch_end(self, epoch, logs={}):
            iterations = epoch * len(ds.train_generator) 
            logs.update({'lr': K.eval(self.model.optimizer.lr),                         
                        'no_iter': K.eval(iterations),
                        'w_gamma': K.eval(config['gamma'])})
            
            def _get_img(x, y_gt, idx, epoch):
                images = x
                fixed = images[:,0,:,:,:].astype(np.float32)
                moving = images[:,1,:,:,:].astype(np.float32)
                flow_gt = y_gt[0,...]
                fixed = tf.convert_to_tensor(fixed, dtype = tf.float32)
                images_2 = tf.convert_to_tensor(moving, dtype = tf.float32)
                y_est = model([fixed, moving])
                flow = y_est[0]
                min0 = np.min(np.array([np.min(flow[0,:,:,0]), np.min(flow_gt[:,:,0])]))
                max0 = np.max(np.array([np.max(flow[0,:,:,0]), np.max(flow_gt[:,:,0])]))
                min1 = np.min(np.array([np.min(flow[0,:,:,1]), np.min(flow_gt[:,:,1])]))
                max1 = np.max(np.array([np.max(flow[0,:,:,1]), np.max(flow_gt[:,:,1])]))              
                
                figure, axs = plt.subplots(3, 4, figsize=(6, 9))
                figure.set_figwidth(15)                
                figure.suptitle('Epoch: {0}, img:{1}'.format(epoch, idx[0]))
                
                ## PLOT IMAGES AND HSV FLOWS ##
                axs[0,0].title.set_text('Fixed image')
                axs[0,0].axis('off')
                axs[0,0].imshow(fixed[0,:,:,0], cmap='gray')

                axs[0,1].title.set_text('Moving image')
                axs[0,1].axis('off')
                axs[0,1].imshow(moving[0,:,:,0], cmap='gray') 
                
                axs[0,2].title.set_text('Estimation')
                axs[0,2].axis('off')
                axs[0,2].imshow(tf_Utils.draw_hsv(flow)[0,...])   
                
                axs[0,3].title.set_text('Ground truth')
                axs[0,3].axis('off')
                axs[0,3].imshow(tf_Utils.draw_hsv(flow_gt[None,...])[0,...])
                
                ## PLOT FLOWS ##
                ax5 = axs[1,0].imshow(flow[0,:,:,0], vmin=min0, vmax=max0, cmap='jet', aspect='auto')
                axs[1,0].axis('off')
                figure.colorbar(ax5, ax=axs[1,0])

                ax6 = axs[1,1].imshow(flow_gt[:,:,0], vmin=min0, vmax=max0, cmap='jet', aspect='auto')
                axs[1,1].axis('off')
                figure.colorbar(ax6, ax=axs[1,1])
    
                ax7 = axs[1,2].imshow(flow[0,:,:,1], vmin=min1, vmax=max1, cmap='jet', aspect='auto')
                axs[1,2].axis('off')
                figure.colorbar(ax7, ax=axs[1,2])

                ax8 = axs[1,3].imshow(flow_gt[:,:,1], vmin=min1, vmax=max1, cmap='jet', aspect='auto')
                axs[1,3].axis('off')
                figure.colorbar(ax8, ax=axs[1,3])                
                
                ## WARPING AND SIMULARITY ##
                warped = tf_Utils.warp_flow(flow=flow, img=moving)[0,:,:,0]
                score, diff, mse = get_score(fixed[0,:,:,0].numpy(), warped.numpy())                
                axs[2,0].title.set_text('Warped Estimation')
                axs[2,0].imshow(warped, cmap='gray')
                axs[2,0].axis('off')                
                
                axs[2,1].title.set_text(f'Warped ssim: {np.round(score, 3)}, {np.round(mse, 1)}')
                axs[2,1].imshow(diff)
                axs[2,1].axis('off')                
                
                warped_gt = tf_Utils.warp_flow(flow=flow_gt[None,...], img=moving)[0,:,:,0]
                score_gt, diff_gt, mse_gt = get_score(fixed[0,:,:,0].numpy(), warped_gt.numpy())                
                
                axs[2,2].title.set_text('Warped GT')
                axs[2,2].imshow(warped_gt, cmap='gray')
                axs[2,2].axis('off')                
                
                axs[2,3].title.set_text(f'GT ssim: {np.round(score_gt,3)}, {np.round(mse_gt, 1)}')
                axs[2,3].imshow(diff_gt)
                axs[2,3].axis('off')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                # Closing the figure prevents it from being displayed directly inside
                # the notebook.
                plt.close(figure)
                buf.seek(0)
                # Convert PNG buffer to TF image
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                # Add the batch dimension
                image = tf.expand_dims(image, 0)            
                return image
            
            img_train = _get_img(x_train, y_gt_train, idx_train, epoch)
            img_val = _get_img(x_val, y_gt_val, idx_val, epoch)
            
            with file_writer.as_default():               
                tf.summary.image("Train_Estimation", img_train, step=epoch)
                tf.summary.image("Val_Estimation", img_val, step=epoch)
            super().on_epoch_end(epoch, logs)
    
    # Tensorboard
    logdir = os.path.join("logs", log_dir)
    idx_train = ds.train_generator.idxs[0]
    x_train, y_gt_train = ds.train_generator._get_train_samples(idx=[idx_train])
    idx_val = ds.val_generator.idxs[0]
    x_val, y_gt_val = ds.val_generator._get_train_samples(idx=[idx_val])
    file_writer = tf.summary.create_file_writer(logdir + '/img')
    return CustomTensorBoard(log_dir= logdir, 
                       histogram_freq=1,
                       profile_batch = 0,
                       embeddings_freq=0,
                       write_grads=False)
def Ckpt_callback(logdir):
    # ModelCheckpoints
    ckptdir = 'Model/{0}/Checkpoints'.format(logdir)
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    return ModelCheckpoint(os.path.join(ckptdir, 'best_model.h5'),
                             monitor='val_est_flow_epe',                           
                             verbose=0, 
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto',
                             save_freq='epoch')

def Learning_rate_scheduler(config, ds):
    def get_short_lr(iteration):
        if iteration > 6e5:
            return initial_lrate / 16
        elif iteration > 5e5:
            return initial_lrate / 8
        elif iteration > 4e5:
            return initial_lrate / 4
        elif iteration > 3e5:
            return initial_lrate / 2
        else:
            return initial_lrate
        
    def get_long_lr(iteration):
        if iteration > 1e6:
            return initial_lrate / 16
        elif iteration > 0.8e6:
            return initial_lrate / 8
        elif iteration > 0.6e6:
            return initial_lrate / 4
        elif iteration > 0.4e6:
            return initial_lrate / 2
        else:
            return initial_lrate

    def get_fine_lr(iteration):
        if iteration > 0.5e6:
            return initial_lrate / 16
        elif iteration > 0.4e6:
            return initial_lrate / 8
        elif iteration > 0.3e6:
            return initial_lrate / 4
        elif iteration > 0.2e6:
            return initial_lrate / 2
        else:
            return initial_lrate    
    def step_decay(epoch):
        iteration = epoch * interation_per_epoch
        if s_type == 'short':
            return get_short_lr(iteration)
        elif s_type == 'long':
            return get_long_lr(iteration)
        else:
            assert s_type=='fine', s_type
            return get_fine_lr(iteration)        
    
    interation_per_epoch = len(ds.train_generator)    
    initial_lrate = config['lr']
    s_type = config['s_type']
    return LearningRateScheduler(step_decay)

