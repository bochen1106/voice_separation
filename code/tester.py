#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 22:20:25 2019

@author: bochen
"""
import sys
import os
from keras.models import load_model
import h5py
import cPickle as pickle

#%%
filename_model = "../exp/001/model.h5"
model = load_model(filename_model)

#%%

if sys.platform in ["linux", "linux2"]: # on server
    path_data = "../../../data_voice_separation/DSD100"
if sys.platform == "darwin":    # on local mac
    path_data = "../../data/DSD100"
        

path_h5 = os.path.join(path_data, "h5")
filename_data = os.path.join(path_h5, "valid.h5")
filename_info = os.path.join(path_h5, "valid.pickle")

f = h5py.File(filename_data)
info = pickle.load(open(filename_info, "r"))


idx = 3

name, [idx_start, length] = info[idx]
data = f["X"][idx_start: idx_start+length, ...]
x = data[None, None, ...]
x.shape
y = model.predict(x)

f.close()