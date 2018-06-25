# FIXME:
import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

from train_with_offset_HC import MultiScaleLossMaxPool, parse_offsets
import os
import numpy as np
import argparse
import vigra
import h5py
import json
import time
import yaml

from inferno.utils.io_utils import yaml2dict

import vigra
import os
from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_postproc_data

import time

import numpy as np

from inferno.utils.io_utils import yaml2dict


from inferno.io.volumetric.volumetric_utils import parse_data_slice

from long_range_hc.criteria.learned_HC.utils.segm_utils import accumulate_segment_features_vigra

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import \
    FixationAgglomeraterFromSuperpixels
import h5py

project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
# aggl_name = 'fancyOverseg_szRg00_fullB_thresh093_blckws_2'


def agglomerate_blocks(project_folder, aggl_name):
    print("Loading affinities and init. segmentation...")
    affinities, gt = import_dataset(project_folder, aggl_name,
                                data_to_import=['affinities', 'gt'])


    # affinities, init_segm = import_postproc_data(project_folder, aggl_name=aggl_name,
    #                                              data_to_import=['affinities', 'init_segmentation'])

    affs_config = yaml2dict(os.path.join(project_folder, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = affs_config['sample']

    gt_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    slc = tuple(parse_data_slice(affs_config['slicing_config']['data_slice']))
    # # bb = np.s_[65:71]
    bb = np.s_[slc[1:]]
    with h5py.File(gt_path, 'r') as f:
        gt = f['segmentations/groundtruth_fixed'][bb].astype('uint64')

    # affinities, gt = import_dataset(project_folder, aggl_name,
    #                             data_to_import=['affinities', 'gt'])

    finalSegm = import_segmentations(project_folder, aggl_name,
                                     keys_to_return=['finalSegm_blocks'])
    # finalSegm, blocks = import_segmentation(project_folder, aggl_name,return_blocks=True)

    # Set up final agglomerater:

    post_proc_config = yaml2dict(os.path.join(project_folder, "postprocess/{}/main_config.yml".format(aggl_name)))

    n_threads = post_proc_config['nb_threads']
    offsets = affs_config['offsets']

    HC_config = post_proc_config['generalized_HC_kwargs']['final_agglomeration_kwargs']
    HC_config['extra_aggl_kwargs']['threshold'] = 0.5
    # extra_aggl_kwargs: {postponeThresholding: false, sizeRegularizer: 0.0, sizeThresMax: 120.0,
    #                     sizeThreshMin: 0.0, threshold: 0.93}

    final_agglomerater = FixationAgglomeraterFromSuperpixels(
                        offsets,
                        n_threads=n_threads,
                        invert_affinities=post_proc_config.get('invert_affinities', False),
                         **HC_config
            #{ 'zero_init': False,
        # 'max_distance_lifted_edges': 5,
        # 'update_rule_merge': 'mean',
        # 'update_rule_not_merge': 'mean'})
        )

    crop_slice_affs = (slice(None), slice(None), slice(None), slice(None))
    # crop_slice_affs = (slice(None), slice(0,45), slice(0,500), slice(0,500))

    crop_slice_segm = crop_slice_affs[1:]
    tick = time.time()
    print("Computing agglomeration...")
    finalSegm_aggl = final_agglomerater(affinities[crop_slice_affs], finalSegm[crop_slice_segm])
    print("TIME: ", time.time() - tick)

    print("Writing...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(finalSegm_aggl, file_path, 'finalSegm', compression='gzip')

    # ------------- WSDT ------------------
    # from copy import deepcopy
    # from skunkworks.postprocessing.watershed.wsdt import WatershedOnDistanceTransform, WatershedOnDistanceTransformFromAffinities
    # WSDT_kwargs = deepcopy(post_proc_config['generalized_HC_kwargs']['WSDT_kwargs'])
    # fragmenter = WatershedOnDistanceTransformFromAffinities(
    #                 offsets,
    #                     WSDT_kwargs.pop('threshold', 0.5),
    #                     WSDT_kwargs.pop('sigma_seeds', 0.),
    #                     invert_affinities=post_proc_config['invert_affinities'],
    #                     return_hmap=False,
    #                     n_threads=n_threads,
    #                     **WSDT_kwargs,
    #                     **post_proc_config['generalized_HC_kwargs']['prob_map_kwargs'])
    #
    #
    # WSDT_segm = fragmenter(affinities)




    # print(segmToPostproc.max())
    # segmToPostproc = vigra.analysis.labelVolume(segmToPostproc.astype(np.uint32))

    # out = final_agglomerater(affinities,segmToPostproc)

    # crop_slice = (slice(None),slice(270,1198),slice(158,1230))
    #
    print("Computing score...")
    evals = cremi_score(gt[crop_slice_segm], finalSegm_aggl, border_threshold=None, return_all_scores=True)
    print(evals)


for aggl_name in [
    'MWS_stride_1_10_blck2_A',
                  # 'fancyOverseg_betterWeights_fullC_thresh093_blckws',
                  # 'fancyOverseg_betterWeights_fullB_thresh093_blckws_1',
    # 'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2',
]:
    agglomerate_blocks(project_folder,aggl_name)