from PWC_Net import PWC_Net, Tensorboard_callback, Learning_rate_scheduler, Ckpt_callback
from configs import dicom_config
from datetime import datetime
from copy import deepcopy

import sys
from argparse import ArgumentParser

def main(config):
    print(config)
    pwc_net = PWC_Net(config = config, name="PWC_Net")    
    log_dir = "{0}_{1}_{2}_{3}".format(datetime.now().strftime("%Y%m%d-%H%M%S"), 'short', 'dicom', config['use_dense_net'])
    tensorboard = Tensorboard_callback(log_dir, config, pwc_net.ds, pwc_net.model)
    lr_scheduler = Learning_rate_scheduler(pwc_net.config, pwc_net.ds)
    ckpt = Ckpt_callback(log_dir)
    callbacks = [tensorboard, lr_scheduler, ckpt]
    pwc_net.run(callbacks)
    pwc_net.model.save_weights('pwc_model_S{0}.h5'.format(log_dir))

if __name__ == "__main__":     
    parser = ArgumentParser(description='Run PWC-model')
    parser.add_argument('-gpu', help='GPU (cpu, 0, 1, 2, 3 or Any), default:Any', default='Any')
    args = parser.parse_args()
    gpu = args.gpu
    if gpu != "Any":
        import os
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
        # The GPU id to use, usually either "0" or "1";
        gpu = "" if gpu == "cpu" else gpu
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu #0, 1, 2, 3";
        print("Using device {0}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    dicom_config['use_dense_net'] = True
    main(dicom_config)
