'''
this script generate audio segments from the original dataset files
it automatically filter out the silent regions from both vocal and accom
'''


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
    path_dataset = "/Volumes/Bochen_Harddrive/dataset/DSD100"
    path_data = "../../data/DSD100"

path_audio_vocal = os.path.join(path_data, "audio", "vocal")
path_audio_accom = os.path.join(path_data, "audio", "accom")

#%%
if not os.path.exists(path_audio_vocal):
    os.makedirs(path_audio_vocal)
if not os.path.exists(path_audio_accom):
    os.makedirs(path_audio_accom)

piecenames = os.listdir(os.path.join(path_dataset, "sources", "Dev"))
piecenames = [x for x in piecenames if x[0]!="." ]  # remove the "hidden folder"
piecenames.sort()

i_sample = 0
for name in piecenames:
    
    print name
    filename = os.path.join(path_dataset, "Sources", "Dev", name, "vocals.wav")
    wav_vocal, sr = librosa.core.load(filename, SR)
    
    filename = os.path.join(path_dataset, "Sources", "Dev", name, "bass.wav")
    wav_1, sr = librosa.core.load(filename, SR)
    filename = os.path.join(path_dataset, "Sources", "Dev", name, "drums.wav")
    wav_2, sr = librosa.core.load(filename, SR)
    filename = os.path.join(path_dataset, "Sources", "Dev", name, "other.wav")
    wav_3, sr = librosa.core.load(filename, SR)
    wav_accom = wav_1 + wav_2 + wav_3
    
    # segment the wav dynamically to avoid silence
    duration = len(wav_vocal) / sr
    idx_sec_start = 0
    idx_sec_cur = 0
    while idx_sec_cur+1 < duration:
        idx_sec_cur += 1
        seg_vocal = wav_vocal[ (idx_sec_cur-1)*sr : idx_sec_cur*sr ]
        seg_accom = wav_accom[ (idx_sec_cur-1)*sr : idx_sec_cur*sr ]
        if np.sqrt(np.mean(seg_vocal**2)) < 0.005 or np.sqrt(np.mean(seg_accom**2)) < 0.005:
            idx_sec_start = idx_sec_cur
            continue
        
        if idx_sec_start + DUR == idx_sec_cur:
            i_sample += 1
            name_dur = "%03d-%03d" % (idx_sec_start, idx_sec_cur)
            name_out = "%06d"%i_sample + "@" + name[:3] + "@" + name_dur
            
            data = wav_vocal[idx_sec_start*sr : idx_sec_cur*sr]
            filename_out = os.path.join(path_audio_vocal, name_out) + ".wav"
            librosa.output.write_wav(filename_out, data, sr)
            
            data = wav_accom[idx_sec_start*sr : idx_sec_cur*sr]
            filename_out = os.path.join(path_audio_accom, name_out) + ".wav"
            librosa.output.write_wav(filename_out, data, sr)
            
            idx_sec_start = idx_sec_cur
    

    

