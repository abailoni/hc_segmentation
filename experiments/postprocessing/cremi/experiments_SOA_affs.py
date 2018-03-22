from train_with_offset_HC import MultiScaleLossMaxPool, parse_offsets
import os
import numpy as np
import argparse
import vigra
import h5py
import json
import time

from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric.volumetric_utils import parse_data_slice

from skunkworks.datasets.cremi.loaders import RawVolumeWithDefectAugmentation
from skunkworks.inference import SimpleInferenceEngine
from skunkworks.postprocessing import local_affinity_multicut_from_wsdt2d
from skunkworks.postprocessing.watershed import DamWatershed
from skunkworks.postprocessing.pipelines import fixation_agglomerative_clustering_from_wsdt2d
from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import toh5

from cremi.evaluation import NeuronIds
from cremi import Volume

from path import get_template_config_file

from skunkworks.metrics.cremi_score import cremi_score


def evaluate(project_folder, sample, offsets, data_slice,
             n_threads, name_aggl):
    pred_path = os.path.join(project_directory,
                             'Predictions',
                             'prediction_sample%s.h5' % sample)
    parsed_slice = parse_data_slice(data_slice)
    # LOAD DATA:
    print("Load prediction and GT..")
    slice_prediction = [slice(None)] + parsed_slice
    bb = np.s_[tuple(slice_prediction)]
    with h5py.File(pred_path, 'r') as f:
        prediction = f['data'][bb].astype('float32')
    # prediction = vigra.readHDF5(pred_path, 'data')[slice_prediction]
    print(prediction.shape)


    gt_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    # bb = np.s_[65:71]
    bb = np.s_[tuple(parsed_slice)]
    with h5py.File(gt_path, 'r') as f:
        gt = f['segmentations/groundtruth'][bb].astype('uint64')

    print(gt.shape)
    # assert gt.shape == prediction.shape[0]


    # postprocess = local_affinity_multicut_from_wsdt2d(n_threads=12)
    # postprocess = DamWatershed(offsets, stride=[1, 10, 10],
    #              n_threads=8)

    # Fixation clustering:

    train_config = os.path.join(project_folder, 'train_config.yml')
    config = yaml2dict(train_config)
    init_segm_opts = config['HC_config']['init_segm']
    postprocess = fixation_agglomerative_clustering_from_wsdt2d(
        offsets,
        **init_segm_opts['wsdt_kwargs'],
        **init_segm_opts['prob_map_kwargs'],
        n_threads=n_threads,
    probability_long_range_edges=0.7,
    return_fragments=False)

    print("fHC segmentation..")
    tick = time.time()
    # fragments, pred_segm = postprocess(prediction)
    print(prediction.mean())
    pred_segm = postprocess(1.-prediction)
    print("Agglomeration took {} s".format(time.time()-tick))

    name_finalSegm = 'finalSegm' if name_aggl is None else 'finalSegm_' + name_aggl
    name_fragments = 'fragments' if name_aggl is None else 'fragments_' + name_aggl
    vigra.writeHDF5(pred_segm.astype('int64'), pred_path, name_finalSegm, compression='gzip')
    # vigra.writeHDF5(fragments.astype('int64'), pred_path, name_fragments, compression='gzip')
    # FIXME: change this...

    # gt = gt[:2]

    evals = cremi_score(gt, pred_segm, border_threshold=None, return_all_scores=True)
    print(evals)

    eval_file = os.path.join(project_directory, 'evaluation_'+name_aggl+'.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            res = json.load(f)
    else:
        res = {}

    res[sample] = evals
    with open(eval_file, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('offset_file', type=str)
    parser.add_argument('gpu', type=int)
    parser.add_argument('--data_slice', default='85:,:,:')
    parser.add_argument('--n_threads', default=1, type=int)
    parser.add_argument('--name_aggl', default=None)

    args = parser.parse_args()

    project_directory = args.project_directory
    gpu = args.gpu

    offset_file = args.offset_file
    offsets = parse_offsets(offset_file)
    data_slice = args.data_slice
    n_threads = args.n_threads
    name_aggl = args.name_aggl


    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    samples = ('B')

    for sample in samples:
        evaluate(project_directory, sample, offsets, data_slice, n_threads, name_aggl)
