import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

from long_range_hc.datasets.segm_transform import FindBestAgglFromOversegmAndGT
import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations

from skunkworks.metrics.cremi_score import cremi_score

project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'


for aggl_name in [
    # 'fancyOverseg_szRg00_fullA_thresh093_blckws',
    #               'fancyOverseg_szRg00_fullC_thresh093_blckws',
    #               'fancyOverseg_sizeReg00_secondHalfDaataB_blockwise',
                  'fancyOverseg_szRg00_fullB_thresh093_blckws_2']:
    print("Loading segm {}...".format(aggl_name))
    gt = import_dataset(project_folder, aggl_name,
                                         data_to_import=['gt'])

    WS_segm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=['finalSegm_proveAggl'])

    find_best = FindBestAgglFromOversegmAndGT(border_thickness=0,
                                  number_of_threads=8,
                                              break_oversegm_on_GT_borders=True,
                                              undersegm_threshold=8000)

    # crop_slice = (slice(40,55), slice(500,1000), slice(500,1000))
    crop_slice = (slice(None), slice(None), slice(None))

    print("Computing best labels...")
    best_GT = find_best(WS_segm[crop_slice], gt[crop_slice])

    print("Writing results...")


    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(best_GT, file_path, 'finalSegm_newAggl_best_GT', compression='gzip')

    # print("Computing score...")
    # # Get rid of ingore-label in best-agglomeration:
    # best_GT = np.array(
    #     vigra.analysis.labelMultiArray((best_GT).astype(np.uint32)))
    # evals = cremi_score(gt, best_GT, border_threshold=None, return_all_scores=True)
    # print(evals)

