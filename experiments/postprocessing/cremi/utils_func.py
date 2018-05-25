import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

import vigra
from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric.volumetric_utils import parse_data_slice
import numpy as np
import h5py
import os
import json


def import_datasets(proj_dir, aggl_name, import_affs=True):
    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = config_file['sample']
    dataset_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    slc = tuple(parse_data_slice(config_file['data_slice_not_padded']))

    bb_affs = np.s_[slc]
    bb = np.s_[slc[1:]]
    with h5py.File(dataset_path, 'r') as f:
        raw = f['raw'][bb].astype(np.float32) / 255.
    with h5py.File(dataset_path, 'r') as f:
        gt = f['segmentations/groundtruth_fixed'][bb]
    if import_affs:
        with h5py.File(config_file['path'], 'r') as f:
            affinities = f[config_file['path_in_h5_dataset']][bb_affs].astype(np.float32)
        return raw, gt, affinities
    else:
        return raw, gt


def import_segmentation(proj_dir, aggl_name, return_fragments=False, return_blocks=False):
    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = config_file['sample']

    scores_file = os.path.join(proj_dir, "postprocess/{}/scores.json".format(aggl_name))
    with open(scores_file) as f:
        scores = json.load(f)
    print("{0} --> CS: {1:.4f}; AR: {2:.4f}; VI-split: {3:.4f}; VI-merge: {4:.4f}".format(aggl_name,
                                                                                          scores[sample]['cremi-score'],
                                                                                          scores[sample][
                                                                                              'adapted-rand'],
                                                                                          scores[sample]['vi-split'],
                                                                                          scores[sample]['vi-merge']))

    file_path = os.path.join(proj_dir, "postprocess/{}/pred_segm.h5".format(aggl_name))
    if return_fragments:
        return (vigra.readHDF5(file_path, 'finalSegm').astype(np.uint32),
                vigra.readHDF5(file_path, 'fragments').astype(np.uint32))
    elif return_blocks:
        return (vigra.readHDF5(file_path, 'finalSegm').astype(np.uint32),
                vigra.readHDF5(file_path, 'finalSegm_blocks').astype(np.uint32))
    else:
        return vigra.readHDF5(file_path, 'finalSegm').astype(np.uint32)
