import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np
from inferno.utils.io_utils import yaml2dict

from utils_func import import_datasets, import_segmentation

from long_range_hc.criteria.learned_HC.utils.segm_utils import accumulate_segment_features_vigra

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import \
    FixationAgglomeraterFromSuperpixels


project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
aggl_name = 'fancyOverseg_sizeReg00_secondHalfDaataB_blockwise'

def agglomerate_blocks(project_folder, aggl_name):
    raw, gt, affinities = import_datasets(project_folder, aggl_name, import_affs=True)


    finalSegm, blocks = import_segmentation(project_folder, aggl_name,return_blocks=True)

    # Set up final agglomerater:

    post_proc_config = yaml2dict(os.path.join(project_folder, "postprocess/{}/main_config.yml".format(aggl_name)))
    aff_loader_config = yaml2dict(os.path.join(project_folder, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))

    n_threads = post_proc_config['nb_threads']
    offsets = aff_loader_config['offsets']

    final_agglomerater = FixationAgglomeraterFromSuperpixels(
                        offsets,
                        n_threads=n_threads,
                        invert_affinities=post_proc_config.get('invert_affinities', False),
                         **post_proc_config['generalized_HC_kwargs']['final_agglomeration_kwargs']
            #{ 'zero_init': False,
        # 'max_distance_lifted_edges': 5,
        # 'update_rule_merge': 'mean',
        # 'update_rule_not_merge': 'mean'})
        )

    finalSegm = final_agglomerater(affinities, blocks)

    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(finalSegm, file_path, 'finalSegm', compression='gzip')

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
    # evals = cremi_score(gt[crop_slice], finalSegm[crop_slice], border_threshold=None, return_all_scores=True)
    # print(evals)

agglomerate_blocks(project_folder,aggl_name)