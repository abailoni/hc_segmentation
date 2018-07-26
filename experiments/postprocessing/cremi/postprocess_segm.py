import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_postproc_data

from multiprocessing.pool import ThreadPool, Pool
from long_range_hc.postprocessing.WS_growing import SizeThreshAndGrowWithWS

from skunkworks.metrics.cremi_score import cremi_score

SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/MWSoffs_bound2_pyT4'
# project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/MWSoffs_bound2_addedBCE_allBound01'

for aggl_name in [
    # 'fancyOverseg_betterWeights_fullA_thresh093_blckws',
    #               'fancyOverseg_betterWeights_fullC_thresh093_blckws',
    #               'fancyOverseg_betterWeights_fullB_thresh093_blckws_1',
    # 'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2',
    #               'inferName_v1_01_HC098_A',
    #               'inferName_v1_01_HC098_B',
                  'inferName_v1_01_HC098_C',
                  'inferName_v1_01_HC098_B',
                  'inferName_v1_01_HC098_A'
]:
    print("Loading segm {}...".format(aggl_name))
    # affinities, gt = import_dataset(project_folder, aggl_name,
    #                                      data_to_import=['affinities', 'gt'])
    finalSegm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm'])

    affinities = import_postproc_data(project_folder, aggl_name=aggl_name,
                                                 data_to_import=['affinities'],
                                            crop_slice="1:3,:,:,:")


    print(finalSegm.shape)
    print(affinities.shape)



    grower = SizeThreshAndGrowWithWS(size_threshold=1,
                            offsets=np.array([[0, -1, 0], [0, 0, -1]]),
                            apply_WS_growing=True)


    # 1:
    seeds = np.empty_like(finalSegm)
    max_label = 0
    for z in range(finalSegm.shape[0]):
        print(z)
        partial_out = grower(1 - affinities[:,slice(z,z+1)], finalSegm[slice(z,z+1)])
        seeds[slice(z,z+1)] = partial_out + max_label
        max_label += partial_out.max() + 1

    # # 2:
    # pool = ThreadPool(processes=2)
    # print("Computing WS labels...")
    #
    # seeds = pool.starmap(grower,
    #               zip([1 - affinities[:,[z]] for z in range(finalSegm.shape[0])],
    #                   [finalSegm[[z]] for z in range(finalSegm.shape[0])]))
    #
    # pool.close()
    # pool.join()
    # seeds = np.concatenate(seeds)

    # # 3:
    # seeds = grower(1 - affinities, finalSegm)

    print("Writing...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(seeds, file_path, 'finalSegm_WS', compression='gzip')

    # print("Computing score...")
    # evals = cremi_score(gt, seeds, border_threshold=None, return_all_scores=True)
    # print(evals)




