
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


import os
import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import h5py
import cPickle as pickle
import librosa
import sys
import threading
from time import sleep
from util import *
from reader import Reader
import func_model

#%%
import sys
print "#######################"
print "platform: " + sys.platform
print "#######################"


class Trainer(object):
    
    def __init__(self, config, logger):
        
        self.config = config
        self.logger = logger
        logger.log("================================================")
        logger.log("initialize the TRAINER")
        logger.log("================================================")
        
    def load_data(self):
        
        config = self.config
        logger = self.logger
        logger.log("load the data from h5 files")
        
        if sys.platform in ["linux", "linux2"]: # on server
            path_data = "../../../data_voice_separation/DSD100"
        if sys.platform == "darwin":    # on local mac
            path_data = "../../data/DSD100"
            
        set_idx = config.get("set_idx")
        path_set = os.path.join(path_data, set_idx)
        path_h5 = os.path.join(path_data, "h5")
        
        filename_data = os.path.join(path_h5, "train.h5")
        filename_info = os.path.join(path_h5, "train.pickle")
        data_train = Reader(filename_data, filename_info, config=config)
        logger.log("finish loading train data from: " + filename_data)
        logger.log("total number of train data samples: %d" % len(data_train.data_flow))
        
        filename_data = os.path.join(path_h5, "valid.h5")
        filename_info = os.path.join(path_h5, "valid.pickle")
        data_valid = Reader(filename_data, filename_info, config=config)
        logger.log("finish loading valid data from: " + filename_data)
        logger.log("total number of valid data samples: %d" % len(data_valid.data_flow))

        filename_data = os.path.join(path_h5, "valid.h5")
        filename_info = os.path.join(path_h5, "valid.pickle")
        data_test = Reader(filename_data, filename_info, config=config)
        logger.log("finish loading test data from: " + filename_data)
        logger.log("total number of test data samples: %d" % len(data_test.data_flow))
        
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        
        
    def build_model(self):
        
        config = self.config
        logger = self.logger
        logger.log("build the model")
        
        model = func_model.build_dpcl()
        logger.log("model summary:")
        model.summary()
        model.summary(print_fn=lambda x: logger.file.write(x + '\n'))
            
        self.model = model
        logger.log("finish building the model")
        
    def run(self):
        
        def eval_loss(model, data_valid):
            
            loss_hist = []
#            n_sample = len(data.data_flow)
#            batch_per_reader = n_sample / data.batch_size
            batch_size = data_valid.batch_size
            for idx_batch in range(batch_size):
#                print "%d of %d" % (idx_batch, batch_size)
                data, label, names = data_valid.iterate_batch()
                loss = model.test_on_batch(x=[data], y=[label])
                loss = np.sqrt(loss / ((data.shape[1] * 257) ** 2))
#                print loss
                loss_hist.append(loss)
            data_valid.reset()
            return np.mean(loss)
        
        config = self.config
        logger = self.logger
        logger.log("train the model")
        
        path_exp = os.path.join(config.get("path_exp"), config.get("exp_idx"))
        if not os.path.exists(path_exp):
            os.makedirs(path_exp)
        
        filename_model = os.path.join(path_exp, "model") + ".h5"
        
        batch_size = config.get("batch_size")
        num_epoch = config.get("num_epoch")
        num_patience = config.get("num_patience")
        
        data_train = self.data_train
        data_valid = self.data_valid
        data_test = self.data_test
        model = self.model
        
        n_sample = len(data_train.data_flow)
        num_iter_per_epoch = (n_sample // batch_size) + 1
        num_iter_max = num_epoch * num_iter_per_epoch
        valid_freq = max(5, num_iter_per_epoch//5)
        valid_freq = 2
        disp_freq = max(2, num_iter_per_epoch//25)
        
        logger.log("--------------------------------------------------")
        logger.log("training condition:")
        logger.log("--------------------------------------------------")
        logger.log("batch size: %d" % batch_size)
        logger.log("interations per epoch: %d" % num_iter_per_epoch)
        logger.log("number of epoch: %d" % num_epoch)
        logger.log("max number of interations: %d" % num_iter_max)
        logger.log("display train loss every %d interations" % disp_freq)
        logger.log("compute valid loss every %d interations" % valid_freq)
        logger.log("--------------------------------------------------")
        
        
        loss_train_hist = []
        loss_train_hist_ave = []
        loss_valid_best = float("inf")
        loss_valid_hist = []
        n_iter = 1
        
        try:
            while n_iter < num_iter_max:
                
                data, label, names = data_train.iterate_batch()
                loss = model.train_on_batch(x=[data], y=[label])
                loss = np.sqrt(loss / ((data.shape[1] * 257) ** 2))
                loss_train_hist.append(loss)
                loss_train_hist_ave.append(np.mean(loss_train_hist))
                
                if np.mod(n_iter, disp_freq) == 0:
                    logger.log("iter: %d of %d, train loss: %.4f" % 
                               (n_iter, num_iter_max, loss_train_hist_ave[-1]))
                
                if np.mod(n_iter, valid_freq) == 0:
                    logger.log("computing loss on valid data ...")
                    loss_valid = eval_loss(model=model, data_valid=data_valid)
                    loss_valid_hist.append(loss_valid)
                    logger.log("current valid loss: %.4f (best: %.4f)" % (loss_valid, loss_valid_best))
                    
                    if loss_valid < loss_valid_best:
                        loss_valid_best = loss_valid
                        num_iter_max = max(num_iter_max, n_iter + num_patience*num_iter_per_epoch)
                        logger.log( "saving the model at iter: %d" % n_iter )
                        model.save(filename_model)
                        logger.log( "model saved as %s" % filename_model )
                n_iter += 1
                
        except KeyboardInterrupt:
            logger.log("Training interrupted ...")
            
        data_train.close()
        data_valid.close()
        data_test.close()
        
        logger.log("training is done")
        logger.log("================================================")
        
#%%
from util.config import Config
from util.logger import Logger


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        exp_idx = sys.argv[1]  
    else:
        exp_idx = "001"
    
    filename_config = "../config/config_" + exp_idx + ".json"
        
    config = Config(filename_config)
    config.set("exp_idx", exp_idx)
    
    filename_log = os.path.join(config.get("path_exp"), exp_idx, "log.txt")
    logger = Logger(filename_log)
    
    t = Trainer(config, logger)
    t.build_model()
    t.model.save("tmp.h5")
#    t.load_data()
#    t.run()
    

