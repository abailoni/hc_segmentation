import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os
import json
import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_postproc_data, import_SOA_datasets

from long_range_hc.datasets.path import get_template_config_file, parse_offsets, adapt_configs_to_model
from long_range_hc.criteria.learned_HC.utils.segm_utils import accumulate_segment_features_vigra, map_features_to_label_array, cantor_pairing_fct

from multiprocessing.pool import ThreadPool, Pool

from long_range_hc.postprocessing.segmentation_pipelines.features import save_edge_indicators, save_edge_indicators_students

from skunkworks.metrics.cremi_score import cremi_score

SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
# project_folder = '/export/home/abailoni/learnedHC/model_090_v2/unstrInitSegm_pureDICE'
# project_folder = '/export/home/abailoni/learnedHC/model_050_A_v3/pureDICE_wholeDtSet'
increment_labels = False

key = 'finalSegm_WS'
aggl_name_partial = 'inferName_v100k_repAttrHC095_'
# aggl_name_partial = 'inferName_v100k-alignedTestOversegmPlusMC_HC065_'

project_folder_affs = '/export/home/abailoni/learnedHC/mergeSpCNN/pureDICE_v2'
aggl_name_partial_affs = 'inferName_v100k_signedHC050_'

offsets = parse_offsets(
    '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/offsets_MWS.json')


max_label = 0
all_edge_data = []
all_cantor_ids = []

for sample in [
    # 'C',
    'B',
    # 'A',
]:

    aggl_name = aggl_name_partial + sample
    out_file = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    print("Loading segm {}...".format(aggl_name))
    overSegm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=[key])

    if increment_labels:
        vigra.writeHDF5(overSegm, out_file, key+'_OLD', compression='gzip')
        max_label_sample = overSegm.max()
        overSegm += max_label
        print("Max label sample {}: {} ({})".format(sample, max_label_sample, max_label))
        vigra.writeHDF5(overSegm, out_file, key, compression='gzip')
        max_label += max_label_sample

    # These are prob. maps and are inverted later on
    boundary_probs, raw = import_postproc_data(project_folder_affs, aggl_name=aggl_name_partial_affs + sample,
                                                 data_to_import=['affinities', 'raw'],
                                            crop_slice=":,60:120,350:1000,350:1000")
    # ":,60:120,350:1000,350:1000"
    overSegm = overSegm[60:120,350:1000,350:1000]
    overSegm, _, _ = vigra.analysis.relabelConsecutive(overSegm)


    save_path = os.path.join(project_folder, "postprocess/{}/edge_data.h5".format(aggl_name))
    # save_edge_indicators(boundary_probs, overSegm, offsets,
    #                      save_path, n_threads=8,
    #                      invert_affinities=True)

    save_edge_indicators_students(boundary_probs, overSegm, offsets,
                         save_path, n_threads=8,
                         invert_affinities=False)

    vigra.writeHDF5(overSegm, os.path.join(project_folder, "postprocess/{}/segmentation.h5".format(aggl_name)), 'data', compression='gzip')
    vigra.writeHDF5(raw, os.path.join(project_folder, "postprocess/{}/raw.h5".format(aggl_name)), 'data', compression='gzip')
