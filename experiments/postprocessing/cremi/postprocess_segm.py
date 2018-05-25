import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np

from utils_func import import_datasets, import_segmentation

from long_range_hc.criteria.learned_HC.utils.segm_utils import accumulate_segment_features_vigra

from skunkworks.metrics.cremi_score import cremi_score

project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
aggl_name = 'fancyOverseg_sizeRg00_fullC_blockwise'
raw, gt, affinities = import_datasets(project_folder, aggl_name, import_affs=True)

segms = {}

finalSegm, blocks = import_segmentation(project_folder, aggl_name,return_blocks=True)

# crop_slice = (slice(None),slice(270,1198),slice(158,786))
#
# evals = cremi_score(gt[crop_slice], finalSegm[crop_slice], border_threshold=None, return_all_scores=True)
# print(evals)



segmToPostproc = finalSegm

segmToPostproc = vigra.analysis.labelVolume(segmToPostproc.astype(np.uint32))
segmentsSizes = accumulate_segment_features_vigra([segmToPostproc], [segmToPostproc],['Count'], map_to_image=True).squeeze()

sizeMask = segmentsSizes > 6
seeds = ((segmToPostproc+1)*sizeMask).astype(np.uint32)

from skunkworks.postprocessing.util import from_affinities_to_hmap
offsets = np.array([[0, -1, 0], [0, 0, -1]])
hmap = from_affinities_to_hmap(affinities[1:3], offsets)
print(hmap.shape)

# watershedResult = seeds
watershedResult = np.empty_like(seeds)
for z in range(hmap.shape[0]):
    watershedResult[z], _ = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z], method='RegionGrowing')

watershedResult = vigra.analysis.labelVolume(watershedResult.astype(np.uint32))
print (watershedResult.shape)
print ("Max label after seeded WS: ", watershedResult.max())

file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
vigra.writeHDF5(watershedResult, file_path, 'finalSegm_WS', compression='gzip')

crop_slice = (slice(None),slice(270,1198),slice(158,786))

evals = cremi_score(gt[crop_slice], watershedResult[crop_slice], border_threshold=None, return_all_scores=True)
print(evals)




