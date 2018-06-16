import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

from long_range_hc.datasets.segm_transform import FindBestAgglFromOversegmAndGT
import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_SOA_datasets

from skunkworks.metrics.cremi_score import cremi_score

project_folder = '/export/home/abailoni/learnedHC/input_segm/WSDT_DS1'



for aggl_name, sample in zip([
    # 'WSDTplusHC_thrsh090_sampleA',
    #               'thrsh050_cropped_B',
    #               'thrsh050_cropped_A',
    #               'thrsh050',
                  'thrsh050',
                  'thrsh050',
                  # 'fancyOverseg_betterWeights_fullB_thresh093_blckws_1',
    # 'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2',
],
    [
        # 'A',
     'B',
     'C']):
    aggl_name = aggl_name + "_{}".format(sample)
    print("Loading segm {}...".format(aggl_name))

    finalSegm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm'])

    SOA_proj_dir = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'

    best_GT = import_segmentations(SOA_proj_dir, 'WSDTplusHC_thrsh090_sample{}'.format(sample),
                                     keys_to_return=['finalSegm_best_GT'])

    # initial_crop_slice = (slice(5,120), slice(50,-50), slice(50,-50))
    # best_GT = best_GT[initial_crop_slice]

    # find_best = FindBestAgglFromOversegmAndGT(border_thickness=2,
    #                               number_of_threads=8,
    #                                           break_oversegm_on_GT_borders=True,
    #                                           undersegm_rel_threshold=0.85)

    from long_range_hc.criteria.learned_HC.utils.segm_utils_CY import find_split_GT

    # crop_slice = (slice(40,55), slice(500,1000), slice(500,1000))
    crop_slice = (slice(None), slice(None), slice(None))
    split_GT = find_split_GT(finalSegm[crop_slice], best_GT[crop_slice], size_small_segments_rel=0.02)




    print("Writing results...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(split_GT, file_path, 'finalSegm_split_GT', compression='gzip')


