
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)  # use how much percentage of gpu ram
config = tf.ConfigProto(gpu_options=gpu_options)
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


class Trainer(object):
    
    def __init__(self, logger):
        
        self.logger = logger
        logger.log("================================================")
        logger.log("initialize the TRAINER")
        logger.log("================================================")
        
    def load_data(self):
        
        logger = self.logger
        logger.log("load the data from h5 files")
                
        filename_data = os.path.join(path_h5, "train.h5")
        filename_info = os.path.join(path_h5, "train.pickle")
        data_train = Reader(filename_data, filename_info)
        logger.log("finish loading train data from: " + filename_data)
        logger.log("total number of train data samples: %d" % len(data_train.data_flow))
        
        filename_data = os.path.join(path_h5, "valid.h5")
        filename_info = os.path.join(path_h5, "valid.pickle")
        data_valid = Reader(filename_data, filename_info)
        logger.log("finish loading valid data from: " + filename_data)
        logger.log("total number of valid data samples: %d" % len(data_valid.data_flow))
        
        self.data_train = data_train
        self.data_valid = data_valid
        
        
    def build_model(self, filename_pretrain=None):
        
        logger = self.logger
        logger.log("build the model")
        
        if filename_pretrain:
            model = func_model.load_model(filename_pretrain)
            model = func_model.compile_model(model)
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("load model from: %s", filename_pretrain)
            logger.log("######################################")

        else:
            model = func_model.build_dpcl()
            model = func_model.compile_model(model)
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            logger.log("######################################")
            
            logger.log("build model with random initialization")
            
        logger.log("model summary:")
        model.summary()
        model.summary(print_fn=lambda x: logger.file.write(x + '\n'))
    
        self.model = model
        logger.log("finish building the model")
        
    def run(self):
        
        def eval_loss(model, data_valid):
            
            loss_hist = []
            n_sample = len(data_valid.data_flow)
            batch_per_reader = n_sample / batch_size
            for idx_batch in range(batch_per_reader):
                data, label, names = data_valid.iterate_batch()
                loss = model.test_on_batch(x=[data], y=[label])
                loss = np.sqrt(loss / ((data.shape[1] * 257) ** 2))
                loss_hist.append(loss)
            data_valid.reset()
            return np.mean(loss)
        
        logger = self.logger
        logger.log("train the model")
        
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        
        filename_model = os.path.join(path_model, "model_best")
        filename_model = str(filename_model)    # to deal with some bug by Keras
        

        
        data_train = self.data_train
        data_valid = self.data_valid
        model = self.model
        
        n_sample = len(data_train.data_flow)
        num_iter_per_epoch = (n_sample // batch_size) + 1
        num_iter_max = num_epoch * num_iter_per_epoch
        valid_freq = max(10, num_iter_per_epoch//5)
        disp_freq = max(5, num_iter_per_epoch//25)
        save_freq = 100
        
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
                
                if np.mod(n_iter, save_freq) == 0:
                    filename_model_cur = os.path.join(path_model, "model_iter%04d" % n_iter)
                    func_model.save_model(model, filename_model_cur)
                    logger.log( "model saved as %s" % filename_model_cur )
                    
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
                        logger.log( "saving the best model at iter: %d" % n_iter )
                        func_model.save_model(model, filename_model)
                        logger.log( "model saved as %s" % filename_model )
                
                n_iter += 1
                
        except KeyboardInterrupt:
            logger.log("Training interrupted ...")
            
        data_train.close()
        data_valid.close()
        
        logger.log("training is done")
        logger.log("================================================")
        
#%%
from util.logger import Logger

if __name__ == "__main__":
    
    filename_log = os.path.join(path_exp, "log.txt")
    logger = Logger(filename_log, append=True)
               
    t = Trainer(logger)
    t.build_model("../exp/002/model")
    t.load_data()
    t.run()
    

