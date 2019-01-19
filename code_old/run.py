
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



if __name__ == '__main__':

    if len(sys.argv) > 1:
        exp_idx = sys.argv[1]  
    else:
        exp_idx = "003"
    
    filename_config = "../config/config_" + exp_idx + ".json"
        
    config = Config(filename_config)
    config.set("exp_idx", exp_idx)
    
    filename_log = os.path.join(config.get("path_exp"), exp_idx, "log.txt")
    logger = Logger(filename_log, append=True)
    
    t = Trainer(config, logger)
    t.build_model("../exp/002/model")
#    t.model.save("tmp.h5")
    t.load_data()
    t.run()
    