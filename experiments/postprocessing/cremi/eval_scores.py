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

sample = None

def import_datasets(proj_dir, aggl_name, import_affs=False, import_raw=True):
    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    global sample
    sample = config_file['sample']
    dataset_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    slc = tuple(parse_data_slice(config_file['data_slice_not_padded']))

    bb_affs = np.s_[slc]
    bb = np.s_[slc[1:]]
    out = []
    if import_raw:
        with h5py.File(dataset_path, 'r') as f:
            out.append(f['raw'][bb].astype(np.float32) / 255.)
    with h5py.File(dataset_path, 'r') as f:
        out.append(f['segmentations/groundtruth_fixed'][bb])
    if import_affs:
        with h5py.File(config_file['path'], 'r') as f:
            out.append(f[config_file['path_in_h5_dataset']][bb_affs].astype(np.float32))

    out = out[0] if len(out) == 1 else out
    return  out


def import_segmentation(proj_dir, aggl_name, return_fragments=False, return_blocks=False):
    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = config_file['sample']

    # scores_file = os.path.join(proj_dir, "postprocess/{}/scores.json".format(aggl_name))
    # with open(scores_file) as f:
    #     scores = json.load(f)
    # print(scores)
    # print("{0} --> CS: {1:.4f}; AR: {2:.4f}; Split: {3:.4f}; Merge: {4:.4f}".format(aggl_name,
    #                                                                                 scores[sample]['cremi-score'],
    #                                                                                 scores[sample]['adapted-rand'],
    #                                                                                 scores[sample]['vi-split'],
    #                                                                                 scores[sample]['vi-merge']))

    file_path = os.path.join(proj_dir, "postprocess/{}/pred_segm.h5".format(aggl_name))
    if return_fragments:
        return (vigra.readHDF5(file_path, 'finalSegm').astype(np.uint16),
                vigra.readHDF5(file_path, 'fragments').astype(np.uint16))
    elif return_blocks:
        return (vigra.readHDF5(file_path, 'finalSegm').astype(np.uint16),
                vigra.readHDF5(file_path, 'finalSegm_blocks').astype(np.uint16))
    else:
        return vigra.readHDF5(file_path, 'finalSegm').astype(np.uint16)


project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'


def eval_scores(agglo_name):
    gt = import_datasets(project_folder,agglo_name,import_affs=False, import_raw=False)

    segm = import_segmentation(project_folder, agglo_name, return_blocks=False)


    from skunkworks.metrics.cremi_score import cremi_score

    evals = cremi_score(gt, segm, border_threshold=None, return_all_scores=True)
    print(evals)
    ref = {}
    ref[sample] = evals
    scores_file = os.path.join(project_folder, "postprocess/{}/scores.json".format(agglo_name))
    with open(scores_file, 'w') as f:
        json.dump(ref, f, indent=4, sort_keys=True)

# for agglo_name in ['LRHC_C_part2', 'MWS_C_part2', 'LRHC_C_part1', 'MWS_C_part1', 'LRHC_A_part2', 'MWS_A_part2', 'LRHC_A_part1', 'MWS_A_part1']:
for agglo_name in ['LRHC_SR03_LRE01W']:
    print(agglo_name)
    eval_scores(agglo_name)