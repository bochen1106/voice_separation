#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 22:20:25 2019

@author: bochen
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



import sys
import os
import numpy as np
import h5py
import cPickle as pickle
import librosa
from sklearn.cluster import KMeans
from reader import Reader

import func_model
sys.path.append("../../functions")
import func_data


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
class Tester(object):
    
    def __init__(self, logger):
        
        self.logger = logger
        logger.log("================================================")
        logger.log("initialize the TRAINER")
        logger.log("================================================")
    
    def load_data(self, data_type="test"):
        
        logger = self.logger
        logger.log("load the data from h5 files")
        
        filename_data = os.path.join(path_h5, data_type+".h5")
        filename_info = os.path.join(path_h5, data_type+".pickle")
        f = h5py.File(filename_data)
        info = pickle.load(open(filename_info, "r"))
        if norm_type == "glob":
            data_mean = np.array(f["mean_glob"])
            data_mean = data_mean[None, ...]
            data_std = np.array(f["std_glob"])
            data_std = data_std[None, ...]
        elif norm_type == "freq":
            data_mean = np.array(f["mean_freq"])
            data_std = np.array(f["std_freq"])
        else:
            data_mean = 0
            data_std = 1
        f.close()
        logger.log("finish loading data from: " + filename_data)
        self.info = info
        self.data_mean = data_mean
        self.data_std = data_std
 
        logger.log("finish loading test data from: " + filename_data)
        logger.log("total number of test data samples: %d" % len(info))
        


    def load_model(self):
        
        logger = self.logger
        logger.log("load the model")
        
        filename_model = os.path.join(path_model, "model_best")
        model = func_model.load_model(filename_model)
           
        self.model = model
        logger.log("finish loading the model from %s" % filename_model)        
        
    def run(self):
        
        logger = self.logger
        logger.log("test the model")
        
        if not os.path.exists(path_result):
            os.makedirs(path_result)
    
        
        model = self.model
        info = self.info
        data_mean = self.data_mean
        data_std = self.data_std
        
        for idx in range(len(info)):
            name, [idx_start, length] = info[idx]
            filename_feat = os.path.join(path_feat, name+".h5")
            f_feat = h5py.File(filename_feat)
            mag = np.array( f_feat["mag"] )     # F * T
            pha = np.array( f_feat["pha"] )
            mask = np.array( f_feat["mask"] )   # 2 * F * T
            t = mask.T       # T * F * 2
            f_feat.close()
            
            x = (mag.T - data_mean) / data_std
            x = x[None, ...]        # 1 * T * F
            v = model.predict(x)    # 1 * T * F * D
            T = v.shape[1]
            F = v.shape[2]
            D = v.shape[-1]
            
            v = np.reshape(v, newshape=(-1, D))     # TF * D
            
            kmean = KMeans(n_clusters=2, random_state=0).fit(v)
            y = np.concatenate(((1 - kmean.labels_)[:, None], kmean.labels_[:, None]), axis=1)  # TF * 2
            y = np.reshape(y, newshape=(T, F, -1))  # T * F * 2
            y = y * np.sum(t, axis=-1)[..., None]   # T * F * 2
            mask_pred = y.T     # 2 * F * T
  
            filename = os.path.join(path_result, name+"_1.wav")
            wav = func_data.restore_wav(mag, pha, mask_pred[0, ...])
            librosa.output.write_wav(filename, wav, 16000)
            
            filename = os.path.join(path_result, name+"_2.wav")
            wav = func_data.restore_wav(mag, pha, mask_pred[1, ...])
            librosa.output.write_wav(filename, wav, 16000)
            
            self.mask_pred = mask_pred
            self.mask = mask
    

    
'''
a dumb logger object when input logger is None
the dumb logger doesn't output any log file to disk
this is used to prevent overwriting an existing log file
'''
class Logger_dumb(object):
    def __init__(self):
        pass
    def log(self, content):
        print (content)

if __name__ == "__main__":
    
    logger = Logger_dumb()
    t = Tester(logger)
    t.load_data()
    t.load_model()
    t.run()









