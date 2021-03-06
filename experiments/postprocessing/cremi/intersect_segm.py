import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os
from shutil import copyfile

from segmfriends.transform.inferno.temp_crap import FindBestAgglFromOversegmAndGT
import numpy as np
from multiprocessing.pool import ThreadPool, Pool

from segmfriends.io.load import import_segmentations
from segmfriends.io.load import import_SOA_datasets, import_dataset

from skunkworks.metrics.cremi_score import cremi_score

from segmfriends.utils.various import cantor_pairing_fct

# project_folder = '/export/home/abailoni/learnedHC/input_segm/WSDT_DS1'
SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
# project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/MWSoffs_bound2_addedBCE_allBound01'

project_folder_1 = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
aggl_name_partial_1 = 'inferName_v100k-alignedTestDts_HC097plusDTWSplusMWS_'
key_1 = 'finalSegm_WS'
# key_1 = 'finalSegm_WS_inters'

project_folder_2 = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
aggl_name_partial_2 = 'inferName_v100k-alignedTestDts_HC098_'
# aggl_name_partial_2 = 'inferName_v100k_DTWS_'
key_2 = 'finalSegm_WS'
# key_2 = 'finalSegm'

project_folder_out = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
aggl_name_partial_out = 'inferName_v100k-alignedTestDts_HC097plusDTWSplusMWS_'
key_out = 'finalSegm_new'



for sample in [
    'A',
    # 'B',
    # 'C',
]:
    aggl_name_1 = aggl_name_partial_1 + sample
    aggl_name_2 = aggl_name_partial_2 + sample
    aggl_name_out = aggl_name_partial_out + sample


    segm_1 = import_segmentations(project_folder_1, aggl_name_1,
                                             keys_to_return=[key_1])
    segm_1 = vigra.analysis.labelVolume(segm_1.astype('uint32'))

    segm_2 = import_segmentations(project_folder_2, aggl_name_2,
                                  keys_to_return=[key_2])
    segm_2 = vigra.analysis.labelVolume(segm_2.astype('uint32'))

    intersection_segm = cantor_pairing_fct(segm_1, segm_2)
    intersection_segm = vigra.analysis.labelVolume(intersection_segm.astype('uint32'))


    print("Writing results...")
    dir_path = os.path.join(project_folder_out, "postprocess/{}/".format(aggl_name_out))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        orig_dir = os.path.join(project_folder_1, "postprocess/{}/".format(aggl_name_1))
        # FIXME: score is copied instaed of being computed:
        for file_name in ['aff_loader_config.yml', 'main_config.yml', 'scores.json']:
            src = os.path.join(orig_dir, file_name)
            if os.path.exists(src):
                dst = os.path.join(dir_path, file_name)
                copyfile(src, dst)

    file_path = os.path.join(project_folder_out, "postprocess/{}/pred_segm.h5".format(aggl_name_out))
    vigra.writeHDF5(intersection_segm, file_path, key_out, compression='gzip')

    # print("Computing score...")
    # evals = cremi_score(gt, seeds, border_threshold=None, return_all_scores=True)
    # print(evals)
    #
    # eval_file = os.path.join(project_folder, "postprocess/{}/scores.json".format(aggl_name))
    # if os.path.exists(eval_file):
    #     with open(eval_file, 'r') as f:
    #         res = json.load(f)
    # else:
    #     res = {}
    #
    # if 'finalSegm_WS' not in res:
    #     res['finalSegm_WS'] = {}
    # res['finalSegm_WS'][sample] = evals
    # # if given_initSegm:
    # #     res['init_segm'][sample] = initSegm_evals
    # with open(eval_file, 'w') as f:
    #     json.dump(res, f, indent=4, sort_keys=True)



