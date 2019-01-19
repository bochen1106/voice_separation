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
import func_model
from sklearn.cluster import KMeans

#%%


#%%

if sys.platform in ["linux", "linux2"]: # on server
    path_data = "../../../data_voice_separation/DSD100"
if sys.platform == "darwin":    # on local mac
    path_data = "../../data/DSD100"
        

path_h5 = os.path.join(path_data, "h5")
filename_data = os.path.join(path_h5, "valid.h5")
filename_info = os.path.join(path_h5, "valid.pickle")

filename_model = "../exp/002/model"
model = func_model.load_model(filename_model)


f = h5py.File(filename_data)
info = pickle.load(open(filename_info, "r"))
n_sample = len(info)
print n_sample
for idx in range(1):
    name, [idx_start, length] = info[idx]
    data = f["X"][idx_start: idx_start+length, ...]
    mask = f["Y"][idx_start: idx_start+length, ...]
    x = data[None, ...]
    v = model.predict(x)
    T = v.shape[1]
    F = v.shape[2]
    D = v.shape[-1]
    
    v = np.reshape(v, newshape=(-1, D))
    
    kmean = KMeans(n_clusters=2, random_state=0).fit(v)
    y = np.concatenate(((1 - kmean.labels_)[:, None], kmean.labels_[:, None]), axis=1)
    y = np.reshape(y, newshape=(T, F, -1))
    y = y * np.sum(mask, axis=-1)[..., None]
    
f.close()