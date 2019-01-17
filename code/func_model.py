#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:04:16 2018

@author: bochen
"""

from keras import backend as K
import keras.layers as KL
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply
from keras.layers import TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.models import model_from_json

import os.path as osp
import os


NUM_LAYER = 4
DIM_HID = 300
DIM_INPUT = 257
DIM_EMBED = 20
NUM_TRACK = 2


def affinity_kmeans(Y, V):
    '''
    Y:  ground-truth (batch_size * T * d * 2)
    V:  prediction (batch_size * T * d * dim_embed)
    '''
    dim_embed = int(str(V.shape[3]))
    num_track = NUM_TRACK
    
    def norm(tensor):
        """ frobenius norm
        """
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2
    
    def dot(x, y):
        """ batch dot
        :param x : batch_size x emb1 x (T x d)
        :param y : batch_size x (T x d) x emb2
        """
        return K.batch_dot(x, y, axes=[2, 1])
    
    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])
    
    # batch_size * (T * d) * dim_embed
    V = KL.Reshape(target_shape=(-1, dim_embed))(V)
    
    # batch_size * (T * d) * 2
    Y = KL.Reshape(target_shape=(-1, num_track))(Y)
        
    # silence_mask: batch_size * (T * d) * 1
    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V
    
    # return with size: (batch_size, )
    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))



def build_dpcl():
    
    import tensorflow as tf
    num_layer = NUM_LAYER
    dim_hid = DIM_HID
    dim_input = DIM_INPUT
    dim_embed = DIM_EMBED
    
    def l2_norm(inputs):
        import tensorflow as tf
        return tf.nn.l2_normalize(inputs, -1)
    
    # batch_size * T * dim_input
    input_audio = Input(shape=(None, dim_input))
    
    x = input_audio
    for i in range(num_layer):
        x = Bidirectional(LSTM(units=dim_hid, return_sequences=True,
                               implementation=2,
                               recurrent_dropout=0.2))(x)
    
    # batch_size * T * (dim_input * dim_embed)
    embed_audio = TimeDistributed(Dense(dim_input*dim_embed, activation="tanh"),
                                  name="embed_audio")(x)
    
    # batch_size * T * dim_input * dim_embed
    embed_audio = KL.Reshape(target_shape=(-1, dim_input, dim_embed))(embed_audio)
    
    # batch_size * (T * dim_input) * dim_embed
    # and normalize along the last dimension
    embed_audio = KL.Reshape(target_shape=(-1, dim_embed))(embed_audio)
    embed_audio = KL.Lambda(l2_norm)(embed_audio)
    
    # Reshape back to:
    # batch_size * T * dim_input * dim_embed
    embed_audio = KL.Reshape(target_shape=(-1, dim_input, dim_embed))(embed_audio)
    
    model = Model(inputs=[input_audio], outputs=[embed_audio])
    
    model.compile(loss=affinity_kmeans, optimizer=Adam(lr=0.001))
    
    return model



#%%

if __name__ == "__main__":
    
    model = build_dpcl()
    model.summary()





