import sys
import os

if sys.platform in ["linux", "linux2"]: # on server
    path_data = "../../../data_voice_separation/DSD100"
if sys.platform == "darwin":    # on local mac
    path_data = "../../data/DSD100"

set_idx = "set_001"
path_set = os.path.join(path_data, set_idx)
path_audio = os.path.join(path_set, "audio")
path_feat = os.path.join(path_set, "feat")
path_h5 = os.path.join(path_set, "h5")

exp_idx = "002"
path_exp = os.path.join("../exp", exp_idx)
path_model = os.path.join(path_exp, "model")
path_result = os.path.join(path_exp, "result")

sr = 16000
dim_feat = 257
dim_embed = 20
num_frame = 626

norm_type = "glob"
seed = 999
batch_size = 32

num_epoch = 30
num_patience = 10




##

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

