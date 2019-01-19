
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)  # use how much percentage of gpu ram
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


import sys
import os

from util import Config
from util import Logger
from trainer import Trainer

from set_config import * 

# path
path_feat = path_feat
path_h5 = path_h5
path_exp = path_exp
path_model = path_model
path_result = path_result

# parameters for data
dim_feat = dim_feat
dim_embed = dim_embed
num_frame = num_frame

norm_type = norm_type
seed = seed
batch_size = batch_size

# parameters for train
num_epoch = num_epoch
num_patience = num_patience

#%%

if __name__ == '__main__':

    filename_log = os.path.join(path_exp, "log.txt")
    logger = Logger(filename_log, append=False)
    
    path_codeback = os.path.join(path_exp, "codeback")
    if not os.path.exists(path_codeback):
        os.makedirs(path_codeback)
    os.system("cp *.py %s" % path_codeback)
    
    
    t = Trainer(logger)
    t.build_model()
    t.load_data()
    t.run()
    
    
    #%%
    
    
    