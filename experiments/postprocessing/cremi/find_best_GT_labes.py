import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

from long_range_hc.datasets.segm_transform import FindBestAgglFromOversegmAndGT
import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_SOA_datasets

from skunkworks.metrics.cremi_score import cremi_score

# project_folder = '/export/home/abailoni/learnedHC/input_segm/WSDT_DS1'
SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/MWSoffs_bound2_pyT4'


for sample in [
    'A',
    'B',
    'C'
]:
    aggl_name = 'inferName_v1_DTWS_' + sample
    print("Loading segm {}...".format(aggl_name))

    gt = import_SOA_datasets(data_to_import=['gt'], sample=sample)

    WS_segm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm'])

    find_best = FindBestAgglFromOversegmAndGT(border_thickness=2,
                                  number_of_threads=8,
                                              break_oversegm_on_GT_borders=True,
                                              undersegm_rel_threshold=0.80)

    # crop_slice = (slice(40,55), slice(500,1000), slice(500,1000))
    crop_slice = (slice(None), slice(None), slice(None))

    print("Computing best labels...")
    best_GT = find_best(WS_segm[crop_slice], gt[crop_slice])

    print("Writing results...")


    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(best_GT, file_path, 'finalSegm_bestGT', compression='gzip')

    # print("Computing score...")
    # # Get rid of ingore-label in best-agglomeration:
    # best_GT = np.array(
    #     vigra.analysis.labelMultiArray((best_GT).astype(np.uint32)))
    # evals = cremi_score(gt, best_GT, border_threshold=None, return_all_scores=True)
    # print(evals)

