from tensorflow.python.ops import summary_ops_v2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, Callback, LambdaCallback
from CustomLayers.WarpLayer import Warp
import matplotlib.pyplot as plt
import os, math, io, time
import tensorflow.keras.backend as K
import tensorflow as tf

def Tensorboard_callback(log_dir, config, ds, model):
    class CustomTensorBoard(TensorBoard):
        def __init__(self, use_atlas, **kwargs):  # add other arguments to __init__ if you need
            self.use_atlas = use_atlas
            super().__init__(**kwargs)
        
        def on_epoch_end(self, epoch, logs={}):
            iterations = epoch * len(ds.train_generator)             
            
            def image2d(img, axis):
                if axis==0:
                    return img[0, img.shape[1]//2, :, :, 0]
                if axis==1:
                    return img[0, :, img.shape[2]//2, :, 0]
                if axis==2:
                    return img[0, :, :, img.shape[3]//2, 0]
            
            def _plot(axs, use_atlas, fixed, moving, warped, fixed_label, moving_label, warped_label):
                for i in range(3):
                    axs[i*3].title.set_text('Fixed image')
                    axs[i*3].axis('off')
                    axs[i*3].imshow(image2d(fixed, i%3), cmap='gray')                    
                    if use_atlas:
                        axs[i*3].contour(image2d(fixed_label, i%3))
                    
                    axs[i*3 + 1].title.set_text('Moving image')
                    axs[i*3 + 1].axis('off')
                    axs[i*3 + 1].imshow(image2d(moving, i%3), cmap='gray')
                    if use_atlas:
                        axs[i*3 + 1].contour(image2d(moving_label, i%3))
                     
                    axs[i*3 + 2].title.set_text('Warped image')
                    axs[i*3 + 2].axis('off')
                    axs[i*3 + 2].imshow(image2d(warped, i%3), cmap='gray')
                    if use_atlas:
                        axs[i*3 + 2].contour(image2d(warped_label, i%3))
            
            def _get_img(images, labels, idx, epoch):
                fixed = images[:,0,...]
                moving = images[:,1,...]
                inputs = [fixed, moving]

                if self.use_atlas:
                    fixed_label = labels[:,0,...]
                    moving_label = labels[:,1,...]
                    inputs.append(moving_label)
                else:
                    fixed_label = None
                    moving_label = None
                
                out = model(inputs)
                if self.use_atlas:
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
                _plot(axs, use_atlas, fixed, moving, warped, fixed_label, moving_label, warped_label)
                '''
                ## PLOT IMAGES AND HSV FLOWS ##
                axs[0].title.set_text('Fixed image')
                axs[0].axis('off')
                axs[0].imshow(fixed[0,:,:,fixed.shape[3]//2,0], cmap='gray')
                if use_atlas:
                    axs[0].contour(fixed_label[0,:,:,fixed_label.shape[3]//2,0])
                
                axs[1].title.set_text('Moving image')
                axs[1].axis('off')
                axs[1].imshow(moving[0,:,:,moving.shape[3]//2,0], cmap='gray')
                if use_atlas:
                    axs[1].contour(moving_label[0,:,:,moving_label.shape[3]//2,0])
                
                axs[2].title.set_text('Warped')
                axs[2].axis('off')
                axs[2].imshow(warped[0,:,:,warped.shape[3]//2,0], cmap='gray')
                if use_atlas:
                    axs[2].contour(warped_label[0,:,:,warped_label.shape[3]//2,0])
                
                ## PLOT IMAGES AND HSV FLOWS ##
                axs[3].title.set_text('Fixed image')
                axs[3].axis('off')
                axs[3].imshow(fixed[0,:,fixed.shape[2]//2,:,0], cmap='gray')
                if use_atlas:
                    axs[3].contour(fixed_label[0,:,fixed_label.shape[2]//2,:,0])
                
                axs[4].title.set_text('Moving image')
                axs[4].axis('off')
                axs[4].imshow(moving[0,:,fixed.shape[2]//2,:,0], cmap='gray')
                if use_atlas:
                    axs[4].contour(moving_label[0,:,moving_label.shape[2]//2,:,0])
                
                axs[5].title.set_text('Warped')
                axs[5].axis('off')
                axs[5].imshow(warped[0,:,warped.shape[2]//2,:,0], cmap='gray')
                if use_atlas:
                    axs[5].contour(warped_label[0,:,warped_label.shape[2]//2,:,0])
                
                ## PLOT IMAGES AND HSV FLOWS ##
                axs[6].title.set_text('Fixed image')
                axs[6].axis('off')
                axs[6].imshow(fixed[0,fixed_label.shape[1]//2,:,:,0], cmap='gray')
                if use_atlas:
                    axs[6].contour(fixed_label[0,fixed_label.shape[1]//2,:,:,0])
                
                axs[7].title.set_text('Moving image')
                axs[7].axis('off')
                axs[7].imshow(moving[0,moving.shape[1]//2,:,:,0], cmap='gray')
                if use_atlas:
                    axs[7].contour(moving_label[0,moving.shape[1]//2,:,:,0])
                
                axs[8].title.set_text('Warped')
                axs[8].axis('off')
                axs[8].imshow(warped[0,warped.shape[1]//2,:,:,0], cmap='gray')
                if use_atlas:
                    axs[8].contour(warped_label[0,warped.shape[1]//2,:,:,0])
                '''
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
            
            img_train = _get_img(train_imgs, train_labels, idx_train, epoch)
            img_val = _get_img(val_imgs, val_labels, idx_val, epoch)
            
            #logs.update({'lr': K.eval(self.model.optimizer.lr),                         
            #            'dice': K.eval(dice_train),
            #            'mse': K.eval(mse_train)})
            with file_writer.as_default():               
                tf.summary.image("Train_Estimation", img_train, step=epoch)
                tf.summary.image("Val_Estimation", img_val, step=epoch)
            super().on_epoch_end(epoch, logs)
    
    # Tensorboard
    use_atlas = config['use_atlas']
    logdir = os.path.join("logs", log_dir)
    idx_train = ds.train_generator.idxs[0]
    train_imgs, train_labels = ds.train_generator._get_train_samples(idx=[idx_train])
    
    idx_val = ds.val_generator.idxs[0]
    val_imgs, val_labels = ds.val_generator._get_train_samples(idx=[idx_val])
    file_writer = tf.summary.create_file_writer(logdir + '/img')
    return CustomTensorBoard(use_atlas = use_atlas,
                             log_dir= logdir, 
                       histogram_freq=1,
                       profile_batch = 0,
                       embeddings_freq=0,
                       write_grads=False)