import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os
import json
import numpy as np

from segmfriends.io.load import import_postproc_data, import_segmentations

from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS

from skunkworks.metrics.cremi_score import cremi_score

SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
# project_folder = '/export/home/abailoni/learnedHC/model_090_v2/unstrInitSegm_pureDICE'
# project_folder = '/export/home/abailoni/learnedHC/model_050_A_v3/pureDICE_wholeDtSet'



aggl_name_partial_all = ['inferName_v100k_proveGAEC_fewLRE_',
                     'inferName_v100k_proveMWS_fewLRE_',
                     # 'inferName_v100k_greedyFix_subBlock_',
                     # 'inferName_v100k_GAEC_subBlock_',
                     # 'inferName_v100k_greedyFix_onlyFewEdges_subBlock_',
                     # 'inferName_v100k_MEAN_noContr_subBlock_',
                     # 'inferName_v100k_GAEC_onlyFewEdges_subBlock_',
                     # 'inferName_v100k_GAEC_local_subBlock_',
                     # 'inferName_v100k_MAX_MWS_setup_subBlock_',
                     # 'inferName_v100k_MEAN_local_subBlock_',
                     # 'inferName_v100k_MAX_local_subBlock_',
                     # 'inferName_v100k_MAX_subBlock_',
                     # 'inferName_v100k_MEAN_subBlock_'
                     ]
# aggl_name_partial = 'inferName_v100k-alignedTestOversegmPlusMC_HC065_'
for aggl_name_partial in aggl_name_partial_all:
    sample = 'B'
    CROP_SLICE = "20:25, 200:1230, 200:1230"
    aggl_name = aggl_name_partial + sample
    print("Loading segm {}...".format(aggl_name))
    # gt = import_SOA_datasets(data_to_import=['gt'], sample=sample)
    # affinities, gt = import_dataset(project_folder, aggl_name,
    #                                      data_to_import=['affinities', 'gt'])
    finalSegm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm'])
    finalSegm = vigra.analysis.labelVolume(finalSegm.astype(np.uint32))
    # FIXME:
    # affinities = 1 - import_SOA_datasets(data_to_import=['affinities'],
    #                                   crop_slice="1:3,:,:,:",
    #                                  sample=sample)



    affinities, gt = import_postproc_data(project_folder, aggl_name=aggl_name,
                                                 data_to_import=['affinities', 'GT'],
                                            crop_slice="1:3,"+CROP_SLICE)

    print(finalSegm.shape)
    print(affinities.shape)
    print(affinities.mean())



    grower = SizeThreshAndGrowWithWS(size_threshold=40,
                            offsets=np.array([[0, -1, 0], [0, 0, -1]]),
                            apply_WS_growing=True)


    # # # 1:
    # seeds = np.empty_like(finalSegm)
    # max_label = 0
    # for z in range(finalSegm.shape[0]):
    #     print(z)
    #     partial_out = grower(affinities[:,slice(z,z+1)], finalSegm[slice(z,z+1)])
    #     seeds[slice(z,z+1)] = partial_out + max_label
    #     max_label += partial_out.max() + 1

    # # 2:
    # TODO: fix max label
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

    # 3:
    seeds = grower(1 - affinities, finalSegm)

    print("Writing...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(seeds, file_path, 'finalSegm_WS', compression='gzip')

    print("Computing score...")
    evals = cremi_score(gt, seeds, border_threshold=None, return_all_scores=True)
    print(evals)
    #
    #
    eval_file = os.path.join(project_folder, "postprocess/{}/scores.json".format(aggl_name))
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            res = json.load(f)
    else:
        res = {}

    if 'finalSegm_WS' not in res:
        res['finalSegm_WS'] = {}
    res['finalSegm_WS'][sample] = evals
    # if given_initSegm:
    #     res['init_segm'][sample] = initSegm_evals
    with open(eval_file, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)





