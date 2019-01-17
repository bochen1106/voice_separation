
import os
import glob
import numpy as np
import json
import matplotlib.pyplot as plt

import librosa
import sys
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

path_audio_vocal = os.path.join(path_set, "audio", "vocal")
path_audio_accom = os.path.join(path_set, "audio", "accom")
path_audio_mixed = os.path.join(path_set, "audio", "mixed")

#%%
if not os.path.exists(path_audio_mixed):
    os.makedirs(path_audio_mixed)

filenames = glob.glob(path_audio_vocal + "/*.wav")
names = [os.path.basename(x).split(".")[0] for x in filenames]
names.sort()


for name in names:
    
    print name
    filename = os.path.join(path_audio_vocal, name + ".wav")
    wav_vocal, sr = librosa.core.load(filename, SR)
    
    filename = os.path.join(path_audio_accom, name + ".wav")
    wav_accom, sr = librosa.core.load(filename, SR)
    
    wav_mixed = wav_vocal + wav_accom
    filename_out = os.path.join(path_audio_mixed, name + ".wav")
    librosa.output.write_wav(filename_out, wav_mixed, sr)



    

