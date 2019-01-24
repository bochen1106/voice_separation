
# get log
mkdir exp/003
scp bochen@airgpustation.ece.rochester.edu:projects/voice_separation/exp/003/log.txt  exp/003/
scp -r bochen@airgpustation.ece.rochester.edu:projects/voice_separation/exp/003/codeback/  exp/003/



# get audio separation result
scp -r bochen@airgpustation.ece.rochester.edu:projects/voice_separation/exp/003/result/  exp/003/



# get model
mkdir exp/003/model
scp bochen@airgpustation.ece.rochester.edu:projects/voice_separation/exp/003/model/model_best.json  exp/003/model/
scp bochen@airgpustation.ece.rochester.edu:projects/voice_separation/exp/003/model/model_best.h5  exp/003/model/

