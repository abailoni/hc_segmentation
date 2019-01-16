import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np

from segmfriends.io.load import import_dataset, import_segmentations

project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'

name1 = 'fancyOverseg_betterWeights_fullB_thresh093_blckws_1'
name2 = 'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2'
part1 = import_segmentations(project_folder, name1,
                                         keys_to_return=['finalSegm_WS_best_GT'])

part2 = import_segmentations(project_folder, name2,
                             keys_to_return=['finalSegm_WS_best_GT'])

total = np.concatenate((part1, part2[12:]), axis=0)

file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(name1))
vigra.writeHDF5(total, file_path, 'finalSegm_WS_best_GT_full', compression='gzip')

