
if sys.platform in ["linux", "linux2"]: # on server
    path_data = "../../../data_voice_separation/DSD100"
if sys.platform == "darwin":    # on local mac
    path_data = "../../data/DSD100"
    
path_exp = "../exp"
set_idx = "set_001"
dim_feat = 257
num_frame = 626
seed_reader = 999
norm_type = "glob"
batch_size = 32
num_epoch = 20
num_patience = 10




