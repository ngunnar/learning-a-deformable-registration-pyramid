from tensorflow.python.ops import summary_ops_v2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, Callback, LambdaCallback
from CustomLayers.WarpLayer import Warp
import matplotlib.pyplot as plt
import os, math, io, time
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import re
import copy

def Tensorboard_callback(log_dir, config, ds, model):
    class CustomTensorBoard(TensorBoard):
        def __init__(self, **kwargs):  # add other arguments to __init__ if you need
            super().__init__(**kwargs)
        
        def on_epoch_end(self, epoch, logs={}):
            iterations = epoch * len(train_idxs)             
            
            def image2d(img, axis):
                if axis==0:
                    return img[0, img.shape[1]//2, :, :, 0]
                if axis==1:
                    return img[0, :, img.shape[2]//2, :, 0]
                if axis==2:
                    return img[0, :, :, img.shape[3]//2, 0]
            
            def _plot(axs, fixed, moving, warped, fixed_label = None, moving_label = None, warped_label = None):
                for i in range(3):
                    axs[i*3].title.set_text('Fixed image')
                    axs[i*3].axis('off')
                    axs[i*3].imshow(image2d(fixed, i%3), cmap='gray')                    
                    if fixed_label is not None:
                        axs[i*3].contour(image2d(fixed_label, i%3).astype('int'))
                    
                    axs[i*3 + 1].title.set_text('Moving image')
                    axs[i*3 + 1].axis('off')
                    axs[i*3 + 1].imshow(image2d(moving, i%3), cmap='gray')
                    if moving_label is not None:
                        axs[i*3 + 1].contour(image2d(moving_label, i%3).astype('int'))
                     
                    axs[i*3 + 2].title.set_text('Warped image')
                    axs[i*3 + 2].axis('off')
                    axs[i*3 + 2].imshow(image2d(warped, i%3), cmap='gray')
                    if warped_label is not None:
                        axs[i*3 + 2].contour(image2d(warped_label, i%3).numpy().astype('int'))
            
            def _get_img(images, labels, idx, epoch):
                if isinstance(images, list):
                    fixed = images[0][None,...]
                    moving = images[1][None,...]
                else:
                    fixed = images[:,0,...]
                    moving = images[:,1,...]
                inputs = [fixed, moving]

                if labels is not None:
                    if isinstance(labels, list):
                        fixed_label = labels[0][None,...]
                        moving_label = labels[1][None,...]
                    else:
                        fixed_label = labels[:,0,...]
                        moving_label = labels[:,1,...]
                    inputs.append(moving_label)
                else:
                    fixed_label = None
                    moving_label = None
                
                out = model.predict_on_batch(inputs)
                if labels is not None:
                    flow = out[-3]
                    warped = out[-2]
                    warped_label = out[-1]
                else:
                    flow = out[-2]
                    warped = out[-1]
                    warped_label = None
                
                #dice = -binary_dice(fixed_label, warped_label)
                #mse = mse_loss(fixed, warped)
                figure, axs = plt.subplots(3, 3, figsize=(6*3, 9))
                axs = axs.flatten()
                figure.set_figwidth(15)                
                figure.suptitle('Epoch: {0}, img:{1} \n {2}'.format(epoch, idx[0], idx[1]))
                _plot(axs, fixed, moving, warped, fixed_label, moving_label, warped_label)                
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
                return image#, dice, mse
            
            imgs_train = []
            for t_img, t_label, i in train_data:
                imgs_train.append(_get_img(t_img, t_label, i, epoch))
            #img_train = _get_img(train_imgs, train_labels, idx_train, epoch)
            
            if len(ds.val_generator.idxs) > 0:
                imgs_val = []
                for t_img, t_label, i in val_data:
                    imgs_val.append(_get_img(t_img, t_label, i, epoch))
                #img_val = _get_img(val_imgs, val_labels, idx_val, epoch)
            
            #logs.update({'lr': K.eval(self.model.optimizer.lr),                         
            #            'dice': K.eval(dice_train),
            #            'mse': K.eval(mse_train)})
            with file_writer.as_default():
                for i in range(len(imgs_train)):
                    tf.summary.image("Train_Estimation_{0}".format(i+1), imgs_train[i], step=epoch)
                if len(ds.val_generator.idxs) > 0:
                    for i in range(len(imgs_val)):
                        tf.summary.image("Val_Estimation_{0}".format(i+1), imgs_val[i], step=epoch)
                
            super().on_epoch_end(epoch, logs)
    
    # Tensorboard
    logdir = os.path.join("logs", log_dir)
    train_idxs = ds.train_generator.idxs.copy()
    train_idxs.sort(key=lambda x: x[0][0])
    #idx_train = ds.train_generator.idxs[0]
    tasks = np.unique([''.join(re.split('(task_\d+)', i[0][0])[0:2]) for i in train_idxs])
    tasks.sort()
    train_data = []
    for task in tasks:
        for i in train_idxs:
            if i[0][0].find(task) == 0:
                d = list(ds.train_generator._get_train_samples(idx=i))
                d.append(i)
                train_data.append(d)
                break
    #train_imgs, train_labels = ds.train_generator._get_train_samples(idx=[idx_train])
    
    if len(ds.val_generator.idxs) > 0:
        val_idxs = ds.val_generator.idxs.copy()
        val_idxs.sort(key=lambda x: x[0][0])
        tasks = np.unique([''.join(re.split('(task_\d+)', i[0][0])[0:2]) for i in val_idxs])
        tasks.sort()
        val_data = []
        for task in tasks:
            for i in val_idxs:
                if i[0][0].find(task) == 0:
                    d = list(ds.val_generator._get_train_samples(idx=i))
                    d.append(i)
                    val_data.append(d)
                    break
        #idx_val = ds.val_generator.idxs[0]
        #val_imgs, val_labels = ds.val_generator._get_train_samples(idx=[idx_val])
    file_writer = tf.summary.create_file_writer(logdir + '/img')
    with file_writer.as_default():
        tf.summary.text('TrainConfig', tf.convert_to_tensor([[i, str(config[i])] for i in sorted(config)]), step=0)
    return CustomTensorBoard(log_dir= logdir, 
                       histogram_freq=1,
                       profile_batch = 0,
                       embeddings_freq=0,
                       write_grads=False)
