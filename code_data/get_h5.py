
import os
import glob
import numpy as np
import json
import matplotlib.pyplot as plt

import librosa
import sys
import util
import h5py
import cPickle as pickle

print "#######################"
print "platform: " + sys.platform
print "#######################"

SR = 16000
DUR = 5 # sec

if sys.platform in ["linux", "linux2"]: # on server
    path_data = "../../../data"
if sys.platform == "darwin":    # on local mac
    path_data = "../../data/DSD100"

set_idx = "set_002"
path_set = os.path.join(path_data, set_idx)

path_feat = os.path.join(path_set, "feat")
path_h5 = os.path.join(path_set, "h5")

#%%
if not os.path.exists(path_h5):
    os.makedirs(path_h5)

filenames = glob.glob(path_feat + "/*.h5")
names = [os.path.basename(x).split(".")[0] for x in filenames]
names.sort()

names_train = names[:1105]
names_valid = names[1105:1399]


#%% train data

X = []
Y = []
info = {}
idx_start = 0
for name in names_train:
    
    print name
    filename = os.path.join(path_feat, name + ".h5")
    
    f = h5py.File(filename)
    mag = np.array(f["mag"])
    mask = np.array(f["mask"])
    f.close()
    
    mag = mag.T
    mask = np.swapaxes(mask, axis1=0, axis2=2)
    X.append(mag)
    Y.append(mask)
    info[name] = [idx_start, mag.shape[0]]
    idx_start += mag.shape[0]

X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
info = sorted(info.items())

mean_freq = np.mean(X, axis=0)
std_freq = np.std(X, axis=0)
mean_glob = np.mean(X)
std_glob = np.std(X)

filename_h5 = os.path.join(path_h5, "train") + ".h5"
if os.path.exists(filename_h5):
    os.remove(filename_h5)
f = h5py.File(filename_h5)
f["X"] = X
f["Y"] = Y
f["mean_freq"] = mean_freq
f["std_freq"] = std_freq
f["mean_glob"] = mean_glob
f["std_glob"] = std_glob
f.close()

filename_pickle = os.path.join(path_h5, "train") + ".pickle"
pickle.dump(info, open(filename_pickle, "w"))



#%% valid data

X = []
Y = []
info = {}
idx_start = 0
for name in names_valid:
    
    print name
    filename = os.path.join(path_feat, name + ".h5")
    
    f = h5py.File(filename)
    mag = np.array(f["mag"])
    mask = np.array(f["mask"])
    f.close()
    
    mag = mag.T
    mask = np.swapaxes(mask, axis1=0, axis2=2)
    X.append(mag)
    Y.append(mask)
    info[name] = [idx_start, mag.shape[0]]
    idx_start += mag.shape[0]

X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
info = sorted(info.items())

mean_freq = np.mean(X, axis=0)
std_freq = np.std(X, axis=0)
mean_glob = np.mean(X)
std_glob = np.std(X)

filename_h5 = os.path.join(path_h5, "valid") + ".h5"
if os.path.exists(filename_h5):
    os.remove(filename_h5)
f = h5py.File(filename_h5)
f["X"] = X
f["Y"] = Y
f["mean_freq"] = mean_freq
f["std_freq"] = std_freq
f["mean_glob"] = mean_glob
f["std_glob"] = std_glob
f.close()

filename_pickle = os.path.join(path_h5, "valid") + ".pickle"
pickle.dump(info, open(filename_pickle, "w"))