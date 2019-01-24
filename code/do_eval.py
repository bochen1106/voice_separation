#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 19:00:46 2019

@author: bochen
"""
import os
import numpy as np
import librosa
import museval


from set_config import * 

# path
path_feat = path_feat
path_h5 = path_h5
path_exp = path_exp
path_model = path_model
path_result = path_result

# parameters for data
sr = sr
path_audio = path_audio
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

path_vocal = os.path.join(path_audio, 'vocal')
path_accom = os.path.join(path_audio, 'accom')
filename_list = os.path.join(path_h5, "names_valid.txt")

names_valid = [x.strip() for x in open(filename_list, "r").readlines()]
names_valid.sort()

n = len(names_valid)


#name = names_valid[15]
sdr_all = []
for i in range(0,5):
    
    if np.mod(i, 100) == 0:
        print "%d of %d" % (i, n)
    name = names_valid[i]
    filename_vocal = os.path.join(path_vocal, name + '.wav')
    filename_accom = os.path.join(path_accom, name + '.wav')
    wav_vocal, sr = librosa.core.load(filename_vocal, sr=sr)
    wav_accom, sr = librosa.core.load(filename_accom, sr=sr)
    wav_gt = np.concatenate((wav_vocal[None, ...], wav_accom[None, ...]), axis=0)
    
    filename_est_1 = os.path.join(path_result, name + '_1.wav')
    filename_est_2 = os.path.join(path_result, name + '_2.wav')
    wav_est_1, sr = librosa.core.load(filename_est_1, sr=sr)
    wav_est_2, sr = librosa.core.load(filename_est_2, sr=sr)  
    wav_est_mix = wav_est_1 + wav_est_2
    wav_est = np.concatenate((wav_est_1[None, ...], wav_est_2[None, ...]), axis=0)
    wav_est_mix = np.concatenate((wav_est_mix[None, ...], wav_est_mix[None, ...]), axis=0)
    
    sdr, isr, sir, sar, perm = museval.metrics.bss_eval(wav_gt, wav_est, 
                                                        window=np.Inf, hop=0, 
                                                        compute_permutation=True)
    sdr = sdr[ perm.ravel() ].ravel()
#    print perm.ravel()
    sdr0, isr0, sir0, sar0, perm0 = museval.metrics.bss_eval(wav_gt, wav_est_mix, 
                                                        window=np.Inf, hop=0, 
                                                        compute_permutation=True)
    sdr0 = sdr0[ perm0.ravel() ].ravel()
    nsdr = sdr - sdr0
#    print perm0.ravel()
    sdr_all.append(np.concatenate( (sdr, nsdr) ))
    

#%%
#
#filename_result = os.path.join(path_result, "_SDR_python.txt")
#np.savetxt(filename_result, sdr_all, fmt="%.4f")



