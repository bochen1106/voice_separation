
import os
import glob
import numpy as np
import json
import matplotlib.pyplot as plt

import librosa
import sys
import util
import h5py

print "#######################"
print "platform: " + sys.platform
print "#######################"

SR = 16000
DUR = 5 # sec

if sys.platform in ["linux", "linux2"]: # on server
    path_data = "../../../data"
if sys.platform == "darwin":    # on local mac
    path_data = "../../data/DSD100"

path_audio_vocal = os.path.join(path_data, "audio", "vocal")
path_audio_accom = os.path.join(path_data, "audio", "accom")
path_audio_mixed = os.path.join(path_data, "audio", "mixed")
path_feat = os.path.join(path_data, "feat")

#%%
if not os.path.exists(path_feat):
    os.makedirs(path_feat)

filenames = glob.glob(path_audio_vocal + "/*.wav")
names = [os.path.basename(x).split(".")[0] for x in filenames]
names.sort()


for name in names:
    
    print name
    filename = os.path.join(path_audio_vocal, name + ".wav")
    wav_vocal, sr = librosa.core.load(filename, SR)
    
    filename = os.path.join(path_audio_accom, name + ".wav")
    wav_accom, sr = librosa.core.load(filename, SR)
    
    filename = os.path.join(path_audio_mixed, name + ".wav")
    wav_mixed, sr = librosa.core.load(filename, SR)
    
    mag, pha, mask = util.cal_spec_mask(wav_mixed, wav_vocal, wav_accom)
    
    filename_feat = os.path.join(path_feat, name) + ".h5"
    f = h5py.File(filename_feat)
    f["mag"] = mag
    f["pha"] = pha
    f["mask"] = mask
    f.close() 

