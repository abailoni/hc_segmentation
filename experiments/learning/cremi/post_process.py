# FIXME:
import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

from long_range_hc.trainers.learnedHC.visualization import VisualizationCallback

import os
import numpy as np
import argparse
import vigra
import h5py

from long_range_hc.datasets.path import get_template_config_file, parse_offsets
import json
import time
import yaml

from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric.volumetric_utils import parse_data_slice

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_postproc_data

from long_range_hc.datasets import AffinitiesVolumeLoader
from long_range_hc.postprocessing.blockwise_solver import BlockWise
from long_range_hc.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import \
    FixationAgglomeraterFromSuperpixels

from cremi.evaluation import NeuronIds
from cremi import Volume

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline


def evaluate(project_folder, sample, offsets,
             n_threads, name_aggl, name_infer, crop_slice=None):
    pred_path = os.path.join(project_folder,
                             'Predictions',
                             'prediction_sample%s.h5' % sample)

    name_aggl = "{}_{}".format(name_aggl, sample)

    data_config_path = os.path.join(project_folder,
                             'infer_data_config_{}_{}.yml'.format(name_infer, sample))
    if not os.path.isfile(data_config_path):
        data_config_path = os.path.join(project_folder,
                                        'data_config.yml')
        aff_path_in_h5file = 'data'
    else:
        aff_path_in_h5file = name_infer
        name_aggl = 'inferName_{}_{}'.format(name_infer, name_aggl)
    data_config = yaml2dict(data_config_path)

    # TODO: save config files associated to this prediction!
    aff_loader_config = './template_config/post_proc/aff_loader_config.yml'
    aff_loader_config = yaml2dict(aff_loader_config)
    aff_loader_config['volumes']['affinities']['path'] = {sample: pred_path}
    aff_loader_config['volumes']['affinities']['path_in_h5_dataset'] = {sample: aff_path_in_h5file}
    aff_loader_config['volumes']['init_segmentation'] = data_config['volume_config']['init_segmentation']
    aff_loader_config['volumes']['GT'] = data_config['volume_config']['GT']
    aff_loader_config['volumes']['raw'] = data_config['volume_config']['raw']

    aff_loader_config['offsets'] = list(offsets)
    aff_loader_config['sample'] = sample
    if crop_slice is not None:
        aff_loader_config['data_slice'][sample] = crop_slice
    else:
        crop_slice = aff_loader_config['data_slice'][sample]


    post_proc_config = './template_config/post_proc/post_proc_config.yml'
    post_proc_config = yaml2dict(post_proc_config)
    post_proc_config['nb_threads'] = n_threads

    # Create sub-directory and save copy of config files:
    postproc_dir = os.path.join(project_folder, "postprocess")
    if not os.path.exists(postproc_dir ):
        os.mkdir(postproc_dir)
    postproc_dir = os.path.join(postproc_dir, name_aggl)
    if not os.path.exists(postproc_dir ):
        os.mkdir(postproc_dir)
    # Dump config files:
    with open(os.path.join(postproc_dir, 'main_config.yml'), 'w') as f:
        yaml.dump(post_proc_config, f)
    with open(os.path.join(postproc_dir, 'aff_loader_config.yml'), 'w') as f:
        yaml.dump(aff_loader_config, f)

    # aff_loader_config.pop('data_slice_not_padded')
    # parsed_slice = tuple(parse_data_slice(aff_loader_config['slicing_config']['data_slice']))

    # TODO: it would be really nice to avoid the full loading of the dataset...
    print("Loading affinities and init. segmentation...")
    affinities, init_segm = import_postproc_data(project_folder, aggl_name=name_aggl,
                         data_to_import=['affinities', 'init_segmentation'])

    # TODO: improve this
    gt_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    slc = tuple(parse_data_slice(crop_slice))
    # # bb = np.s_[65:71]
    bb = np.s_[slc[1:]]
    with h5py.File(gt_path, 'r') as f:
        gt = f['segmentations/groundtruth_fixed'][bb].astype('uint64')

    agglomerater = get_segmentation_pipeline(
        post_proc_config.get('segm_pipeline_type', 'gen_HC'),
        offsets,
        nb_threads=n_threads,
        invert_affinities=post_proc_config.get('invert_affinities', False),
        return_fragments=False,
        **post_proc_config
    )

    # agglomerater = FixationAgglomeraterFromSuperpixels(
    #                 offsets,
    #                 n_threads=n_threads,
    #                 invert_affinities=post_proc_config.get('invert_affinities', False),
    #                  **post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']
    # )

    print("Starting prediction...")
    tick = time.time()
    init_segm, _, _ = vigra.analysis.relabelConsecutive(init_segm.astype('uint32'))
    pred_segm = agglomerater(affinities, init_segm)
    print("Post-processing took {} s".format(time.time() - tick))
    print("Pred. sahpe: ", pred_segm.shape)
    print("GT shape: ", gt.shape)
    print("Min. GT label: ", gt.min())

    segm_file = os.path.join(postproc_dir, 'pred_segm.h5')
    name_finalSegm = 'finalSegm'
    print("Writing on disk...")
    vigra.writeHDF5(pred_segm.astype('int64'), segm_file, name_finalSegm, compression='gzip')

    # print("Connected components if slice is taken...")
    # gt = vigra.analysis.labelVolumeWithBackground(gt.astype('uint32'))
    # init_segm = vigra.analysis.labelVolumeWithBackground(init_segm.astype('uint32'))
    # pred_segm = vigra.analysis.labelVolumeWithBackground(pred_segm.astype('uint32'))
    # # best_gt = vigra.analysis.labelVolumeWithBackground(best_gt.astype('uint32'))
    # # ignore_mask = best_gt != 0

    print("Evaluating scores...")
    initSegm_evals = cremi_score(gt, init_segm, border_threshold=None, return_all_scores=True)
    print("Score of the oversegm:", initSegm_evals)
    evals = cremi_score(gt, pred_segm, border_threshold=None, return_all_scores=True)
    print("Scores achieved: ", evals)

    eval_file = os.path.join(postproc_dir, 'scores.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            res = json.load(f)
    else:
        res = {}

    res[sample] = evals
    res['init_segm'] = {}
    res['init_segm'][sample] = initSegm_evals
    with open(eval_file, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('offset_file', type=str)
    parser.add_argument('--samples', nargs='+', default=['A', 'B', 'C'], type=str)
    parser.add_argument('--crop_slice', default=None)
    parser.add_argument('--n_threads', default=1, type=int)
    parser.add_argument('--name_aggl', default=None)
    parser.add_argument('--name_infer', default=None)

    args = parser.parse_args()

    project_directory = args.project_directory

    offset_file = args.offset_file
    offsets = parse_offsets(offset_file)
    n_threads = args.n_threads
    name_aggl = args.name_aggl
    name_infer = args.name_infer
    samples = args.samples
    crop_slice = args.crop_slice

    for sample in samples:
        evaluate(project_directory, sample, offsets, n_threads, name_aggl, name_infer,
                 crop_slice)
