import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os
import json
import numpy as np

from segmfriends.io.load import import_postproc_data, import_SOA_datasets, import_dataset, import_segmentations, \
    parse_offsets

from long_range_hc.datasets.path import get_template_config_file, adapt_configs_to_model
from segmfriends.features.mappings import map_features_to_label_array
from segmfriends.utils.various import cantor_pairing_fct
from segmfriends.features.vigra_feat import accumulate_segment_features_vigra

from multiprocessing.pool import ThreadPool, Pool

import nifty.graph.rag as nrag

from segmfriends.io.save import save_edge_indicators, save_edge_indicators_students

from skunkworks.metrics.cremi_score import cremi_score

SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
# project_folder = '/export/home/abailoni/learnedHC/model_090_v2/unstrInitSegm_pureDICE'
# project_folder = '/export/home/abailoni/learnedHC/model_050_A_v3/pureDICE_wholeDtSet'

aggl_name_partial = 'inferName_v100k_MEAN_constr_7_'
# aggl_name_partial = 'inferName_v100k-alignedTestOversegmPlusMC_HC065_'

offsets = parse_offsets(
    '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/offsets_MWS.json')

for sample in [
    # 'C',
    'B',
    # 'A',
]:
    crop_slice_str = ":30,:,:"

    aggl_name = aggl_name_partial + sample
    out_file = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    print("Loading segm {}...".format(aggl_name))
    overSegm, UCM = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm', 'UCM'], crop_slice=crop_slice_str)

    def get_UCM_mask(UCM):
        max_label = UCM.max()
        return UCM == max_label

    UCM_mask = get_UCM_mask(UCM)

    # These are prob. maps and are inverted later on
    # ":,60:120,350:1000,350:1000"
    # ":,10:135,350:1000,350:1000"
    UCM_mask = UCM_mask[...,:3]
    print(UCM_mask.shape, overSegm.shape)

    offsets = offsets[:3]
    nb_threads = 8
    new_labels = nrag.connectedComponentsFromEdgeLabels(overSegm.shape, offsets, UCM_mask.astype('float'), nb_threads)

    vigra.writeHDF5(new_labels, os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name)), 'connComponents', compression='gzip')
