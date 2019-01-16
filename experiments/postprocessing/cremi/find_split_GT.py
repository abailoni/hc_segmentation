import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

from segmfriends.transform.inferno.temp_crap import FindBestAgglFromOversegmAndGT, FindSplitGT
import numpy as np

from segmfriends.io.load import import_SOA_datasets, import_dataset, import_segmentations

from skunkworks.metrics.cremi_score import cremi_score

project_folder = '/export/home/abailoni/learnedHC/input_segm/WSDT_DS1'



for aggl_name, sample in zip([
    # 'WSDTplusHC_thrsh090_sampleA',
    #               'thrsh050_cropped_B',
    #               'thrsh050_cropped_A',
                  'thrsh010',
                  # 'thrsh010',
                  # 'thrsh010',
                  # 'fancyOverseg_betterWeights_fullB_thresh093_blckws_1',
    # 'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2',
],
    [
     'C',
        # 'A',
     # 'B',
    ]):
    aggl_name = aggl_name + "_{}".format(sample)
    print("Loading segm {}...".format(aggl_name))

    finalSegm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm'])

    SOA_proj_dir = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'

    # best_GT = import_segmentations(SOA_proj_dir, 'WSDTplusHC_thrsh090_sample{}'.format(sample),
    #                                  keys_to_return=['finalSegm_best_GT'])
    best_GT = import_SOA_datasets(proj_dir=project_folder, aggl_name=aggl_name,
                             data_to_import=['gt'])

    # initial_crop_slice = (slice(5,120), slice(50,-50), slice(50,-50))
    # best_GT = best_GT[initial_crop_slice]

    # Ignore segments smaller than 6%:
    find_splits = FindSplitGT(size_small_segments_rel=0.005,
                             border_thickness_segm=0,
                            border_thickness_GT=0,
                                  number_of_threads=8,
                                              break_oversegm_on_GT_borders=True)



    # crop_slice = (slice(40,55), slice(500,1000), slice(500,1000))
    crop_slice = (slice(0,25), slice(None), slice(None))
    # crop_slice = (slice(None), slice(None), slice(None))
    split_GT = find_splits(finalSegm[crop_slice], best_GT[crop_slice])




    print("Writing results...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(split_GT, file_path, 'finalSegm_split_realGT_prove', compression='gzip')


