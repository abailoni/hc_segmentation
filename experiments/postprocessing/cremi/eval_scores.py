import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

from segmfriends.io.load import import_SOA_datasets, import_dataset, import_segmentations

import matplotlib
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
cmap_random = matplotlib.colors.ListedColormap(np.random.rand(100000, 3))

import os
import vigra
import h5py
from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric.volumetric_utils import parse_data_slice
import json

project_folder = '/export/home/abailoni/learnedHC/model_050_A_v2/pureDICE'

aggl_name_partial = 'inferName_v40k_HC_050_'

for sample in [
    'C',
    # 'B',
    # 'A',
]:
    aggl_name = aggl_name_partial + sample
    gt = import_SOA_datasets(data_to_import=['gt'], sample=sample)

    segm = import_segmentations(project_folder, aggl_name,
                                   keys_to_return=['finalSegm'])

    from skunkworks.metrics.cremi_score import cremi_score

    evals = cremi_score(gt, segm, border_threshold=None, return_all_scores=True)
    print(evals)
    ref = {}
    ref[sample] = evals
    scores_file = os.path.join(project_folder, "postprocess/{}/scores.json".format(aggl_name))
    # with open(scores_file, 'w') as f:
    #     json.dump(ref, f, indent=4, sort_keys=True)
