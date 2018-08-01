import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

from long_range_hc.datasets.segm_transform import FindBestAgglFromOversegmAndGT
import numpy as np
from multiprocessing.pool import ThreadPool, Pool

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_SOA_datasets

from skunkworks.metrics.cremi_score import cremi_score

# project_folder = '/export/home/abailoni/learnedHC/input_segm/WSDT_DS1'
SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
# project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
# project_folder = '/export/home/abailoni/learnedHC/model_050_A_v2/pureDICE'
project_folder = '/export/home/abailoni/learnedHC/model_090_v2/unstrInitSegm_pureDICE'

aggl_name_partial = 'inferName_v1_HC_090_'

for sample in [
    'C',
    'B',
    'A',
]:
    aggl_name = aggl_name_partial + sample
    print("Loading segm {}...".format(aggl_name))

    gt = import_SOA_datasets(data_to_import=['gt'], sample=sample)

    WS_segm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm_WS'])

    find_best = FindBestAgglFromOversegmAndGT(border_thickness=2,
                                  number_of_threads=8,
                                              break_oversegm_on_GT_borders=True,
                                              undersegm_rel_threshold=0.80)


    # crop_slice = (slice(40,55), slice(500,1000), slice(500,1000))
    crop_slice = (slice(None), slice(None), slice(None))

    print("Computing best labels...")

    # # 1: single thread
    best_GT = find_best(WS_segm[crop_slice], gt[crop_slice])

    # # Multithread (analyze single slices...!):
    # pool = ThreadPool(processes=4)
    # print("Computing WS labels...")
    #
    # best_GT = pool.starmap(find_best,
    #               zip([WS_segm[[z]] for z in range(WS_segm.shape[0])],
    #                   [gt[[z]] for z in range(WS_segm.shape[0])]))
    #
    # pool.close()
    # pool.join()
    # # Do final coloring:
    # print("Combine slices: ")
    # # FIXME: overflow int...?
    # max_label = 0
    # for z in range(WS_segm.shape[0]):
    #     best_GT[z] += max_label
    #     max_label += best_GT[z].max() + 1
    # best_GT = np.concatenate(best_GT)
    # best_GT = find_best(best_GT, gt)


    print("Writing results...")


    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(best_GT, file_path, 'finalSegm_WS_bestGT', compression='gzip')

    # print("Computing score...")
    # # Get rid of ingore-label in best-agglomeration:
    # best_GT = np.array(
    #     vigra.analysis.labelMultiArray((best_GT).astype(np.uint32)))
    # evals = cremi_score(gt, best_GT, border_threshold=None, return_all_scores=True)
    # print(evals)

