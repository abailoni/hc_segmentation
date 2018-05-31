import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations

from skunkworks.metrics.cremi_score import cremi_score

project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'


for aggl_name in [
    # 'fancyOverseg_szRg00_fullA_thresh093_blckws',
    #               'fancyOverseg_szRg00_fullC_thresh093_blckws',
    #               'fancyOverseg_sizeReg00_secondHalfDaataB_blockwise',
    'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2',
                  # 'fancyOverseg_szRg00_fullB_thresh093_blckws_2'
]:
    print("Loading segm {}...".format(aggl_name))
    affinities, gt = import_dataset(project_folder, aggl_name,
                                         data_to_import=['affinities', 'gt'])

    finalSegm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm'])



    from long_range_hc.postprocessing.WS_growing import SizeThreshAndGrowWithWS

    grower = SizeThreshAndGrowWithWS(size_threshold=6,
                            offsets=np.array([[0, -1, 0], [0, 0, -1]]),
                            apply_WS_growing=True)

    print("Computing WS labels...")
    seeds = grower(affinities[1:3], finalSegm)

    print("Writing...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(seeds, file_path, 'finalSegm_WS', compression='gzip')

    print("Computing score...")
    evals = cremi_score(gt, seeds, border_threshold=None, return_all_scores=True)
    print(evals)


# # crop_slice = (slice(None),slice(270,1198),slice(158,1230))
# #
#
#
#
# for aggl_name in [
#     'fancyOverseg_sizeReg00_firstHalfDaataB_blockwise',
#                   'fancyOverseg_sizeRg00_fullC_blockwise',
#                   'fancyOverseg_sizeRg00_fullA_blockwise'
#                   ]:
#
#     seeds = import_segmentations(project_folder, aggl_name,
#                                              keys_to_return=['finalSegm_seeds'])
#
#
#     if aggl_name == 'fancyOverseg_sizeReg00_firstHalfDaataB_blockwise':
#         seeds_2 = import_segmentations(project_folder, 'fancyOverseg_sizeReg00_secondHalfDaataB_blockwise',
#                                              keys_to_return=['finalSegm_seeds'])
#         seeds = np.concatenate((seeds, seeds_2[12:]), axis=0)
#         print(seeds.shape)
#
#     print("Loaded segm: ", aggl_name)
#
#     padded_seeds = np.pad(seeds, pad_width=((3,3), (50,50), (50,50)), mode='constant')
#     print(padded_seeds.shape)
#
#     file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
#     vigra.writeHDF5(padded_seeds, file_path, 'finalSegm_seeds_padded', compression='gzip')
#     print("Done!")
