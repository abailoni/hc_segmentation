import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os
from shutil import copyfile



from long_range_hc.datasets.segm_transform import FindBestAgglFromOversegmAndGT
import numpy as np
from multiprocessing.pool import ThreadPool, Pool

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_SOA_datasets

from skunkworks.metrics.cremi_score import cremi_score

from long_range_hc.criteria.learned_HC.utils.segm_utils import cantor_pairing_fct


project_folder = '/export/home/abailoni/learnedHC/model_050_A_v3/pureDICE_wholeDtSet'


name = 'inferName_v100k-alignedTestOversegmPlusMC_MC060_'
# name = 'inferName_v100k-alignedTestDts_MC092_'


oversegm_proj_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
# oversegm_name = 'inferName_v100k-alignedTestDts_alignedTestOversegmPlusMC_'
oversegm_name = 'inferName_v100k-alignedTestDts_HC097plusDTWSplusMWS_'
oversegm_key = 'finalSegm_WS'




# project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
#
# name = 'inferName_v100k-alignedTestDts_alignedTestOversegmPlusMC_'
#
#
# oversegm_proj_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
# oversegm_name = 'inferName_v100k-alignedTestDts_alignedTestOversegmPlusMC_'
# oversegm_key = 'finalSegm'

defected_slices_sp = {
    'A': [10, 43, 61, (89, 90), (118,119)],
    'B': [(25,26), (54,55)],
    'C': [24, 84, 96]
}



for sample in [
    # 'A',
    # 'B',
    'C',
]:

    defected_slices_sample = defected_slices_sp[sample]
    nb_parts = len(defected_slices_sample) + 1

    oversegm = import_segmentations(oversegm_proj_folder, oversegm_name + sample,
                                  keys_to_return=[oversegm_key])

    combined_segm = []

    max_label = 0

    partial_segm = import_segmentations(project_folder, '{}part{}_{}'.format(name, 1, sample),
                                        keys_to_return=['finalSegm'])
    print(1, partial_segm.shape)
    combined_segm.append(partial_segm + max_label)
    max_label += partial_segm.max()

    for i, part in enumerate(range(2, nb_parts+1)):
        if isinstance(defected_slices_sample[i], tuple):
            z_slice_1 = oversegm[[defected_slices_sample[i][0]]]
            z_slice_2 = oversegm[[defected_slices_sample[i][1]]]
            combined_segm.append(z_slice_1 + max_label)
            max_label += z_slice_1.max()
            combined_segm.append(z_slice_2 + max_label)
            max_label += z_slice_2.max()
        else:
            z_slice_1 = oversegm[[defected_slices_sample[i]]]
            combined_segm.append(z_slice_1 + max_label)
            max_label += z_slice_1.max()

        partial_segm = import_segmentations(project_folder, '{}part{}_{}'.format(name, part, sample),
                                  keys_to_return=['finalSegm'])
        print(part, partial_segm.shape)
        combined_segm.append(partial_segm + max_label)
        max_label += partial_segm.max()

    combined_segm = np.concatenate(combined_segm, axis=0)
    print("Relabel continuous...")
    intersection_segm, _, _ = vigra.analysis.relabelConsecutive(combined_segm.astype('uint32'))


    print("Writing results...")
    aggl_name_out = "{}combined_{}".format(name, sample)
    dir_path = os.path.join(project_folder, "postprocess/{}/".format(aggl_name_out))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        orig_dir = os.path.join(project_folder, "postprocess/{}part1_{}".format(name, sample))
        # FIXME: score is copied instaed of being computed:
        for file_name in ['aff_loader_config.yml', 'main_config.yml', 'scores.json']:
            src = os.path.join(orig_dir, file_name)
            if os.path.exists(src):
                dst = os.path.join(dir_path, file_name)
                copyfile(src, dst)

    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name_out))
    vigra.writeHDF5(intersection_segm, file_path, 'finalSegm', compression='gzip')

