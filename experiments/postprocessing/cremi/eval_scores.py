import matplotlib
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
cmap_random = matplotlib.colors.ListedColormap(np.random.rand(100000, 3))

import os
import vigra
import h5py


sample = 'B'
model_name = 'SOA_affinities'
# slice_window = (slice(48,55), slice(560,760), slice(1400,1700))
# slice_window = (slice(30,55), slice(560,860), slice(1400,1700))

# Debug block
slice_window = (slice(28,38), slice(460,860), slice(1200,1650))

# Larger block:
slice_window = (slice(28,78), slice(460,860), slice(1200,1650))

# Even larger block:
slice_window = (slice(28,98), slice(460,1100), slice(900,1650))

# Almost full dataset:
slice_window = (slice(None), slice(197,1350), slice(700,1800))

project_folder = '/export/home/abailoni/learnedHC/new_experiments/'
file_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
predictions_folder = os.path.join(project_folder, model_name)

# Predictions:
predictions_path = os.path.join(predictions_folder, 'Predictions/prediction_sample%s.h5' %(sample))



bb = np.s_[slice_window]
bb_affs = np.s_[(slice(None),) + slice_window]
# with h5py.File(file_path, 'r') as f:
#     raw = f['raw'][bb].astype(np.float32) / 255.
with h5py.File(file_path, 'r') as f:
    gt = f['segmentations/groundtruth'][bb].astype(np.uint16)
# with h5py.File(predictions_path, 'r') as f:
#     affinities = f['data'][bb_affs].astype(np.float32)



from skunkworks.metrics.cremi_score import cremi_score

blocks = vigra.readHDF5(predictions_path, 'finalSegm_mean_weightedLRangeedges_TRY_blocks').astype(np.uint16)
mean_blockwise = vigra.readHDF5(predictions_path, 'finalSegm_mean_weightedLRangeedges_TRY').astype(np.uint16)
WSDT = vigra.readHDF5(predictions_path, 'finalSegm_WSDT_full_dataset').astype(np.uint16)
WSDT_local = vigra.readHDF5(predictions_path, 'finalSegm_WSDT_local_full_dataset').astype(np.uint16)

crop = (slice(60,None),slice(None), slice(140, None))

# print("WSDT naive LR")
# evals = cremi_score(gt[crop], WSDT[crop], border_threshold=None, return_all_scores=True)
# print(evals)
#
# print("WSDT local edges")
# evals = cremi_score(gt[crop], WSDT_local[crop], border_threshold=None, return_all_scores=True)
# print(evals)
#
# print("Smart blockwise")
# evals = cremi_score(gt[crop], mean_blockwise[crop], border_threshold=None, return_all_scores=True)
# print(evals)

print("Blocks:")
evals = cremi_score(gt[crop], blocks[crop], border_threshold=None, return_all_scores=True)
print(evals)