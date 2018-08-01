# FIXME:
import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

from long_range_hc.trainers.learnedHC.visualization import VisualizationCallback

from shutil import copyfile

import os
import numpy as np
import argparse
import vigra
import h5py

from long_range_hc.datasets.path import get_template_config_file, parse_offsets, adapt_configs_to_model
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

from long_range_hc.datasets import AffinitiesHDF5VolumeLoader, AffinitiesVolumeLoader

from cremi.evaluation import NeuronIds
from cremi import Volume

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline


def evaluate(project_folder, sample, offsets,
             n_threads, name_aggl, name_infer, crop_slice=None,
             affinities=None, use_default_postproc_config=False,
             model_IDs=None):
    pred_path = os.path.join(project_folder,
                             'Predictions',
                             'prediction_sample%s.h5' % sample)


    # ------
    # Load data config:
    # ------
    name_aggl = "{}_{}".format(name_aggl, sample)
    data_config_path = os.path.join(project_folder,
                                    'infer_data_config_{}_{}.yml'.format(name_infer, sample))
    if not os.path.isfile(data_config_path):
        raise DeprecationWarning()
        data_config_path = os.path.join(project_folder,
                                        'data_config.yml')
        aff_path_in_h5file = 'data'
    else:
        aff_path_in_h5file = name_infer
        name_aggl = 'inferName_{}_{}'.format(name_infer, name_aggl)
    data_config = yaml2dict(data_config_path)


    # Create sub-directory and save copy of config files:
    postproc_dir = os.path.join(project_folder, "postprocess")
    if not os.path.exists(postproc_dir):
        os.mkdir(postproc_dir)
    postproc_dir = os.path.join(postproc_dir, name_aggl)
    if not os.path.exists(postproc_dir):
        os.mkdir(postproc_dir)


    # Load default postproc config:
    def_postproc_config_path = './template_config/post_proc/post_proc_config.yml'
    postproc_config_path = os.path.join(postproc_dir, 'main_config.yml')
    copyfile(def_postproc_config_path, postproc_config_path)
    post_proc_config = yaml2dict(postproc_config_path)

    # Uptdate volume specs::
    assert 'volumes' in post_proc_config, "Updated: please move affinity loading to post_proc_config!"
    assert 'data_slice' in post_proc_config, "Updated: please move crop_slice to post_proc_config!"

    post_proc_config['volumes']['affinities']['path'] = {sample: pred_path}
    post_proc_config['volumes']['affinities']['path_in_h5_dataset'] = {sample: aff_path_in_h5file}
    given_initSegm = post_proc_config['start_from_given_segm']


    if 'init_segmentation' in data_config['volume_config']:
        post_proc_config['volumes']['init_segmentation'] = data_config['volume_config']['init_segmentation']
    post_proc_config['volumes']['GT'] = data_config['volume_config']['GT']
    post_proc_config['volumes']['raw'] = data_config['volume_config']['raw']

    post_proc_config['offsets'] = list(offsets)
    post_proc_config['sample'] = sample

    post_proc_config['nb_threads'] = n_threads

    with open(postproc_config_path, 'w') as f:
        yaml.dump(post_proc_config, f)

    # Adapt config to the passed model options:
    if model_IDs is not None:
        config_paths = {'models': './template_config/models_config.yml',
                        'postproc': postproc_config_path}
        adapt_configs_to_model(model_IDs, debug=True, **config_paths)
        post_proc_config = yaml2dict(postproc_config_path)

    # Final checks:
    if given_initSegm:
        assert 'init_segmentation' in post_proc_config['volumes'], "Init. segmentation required! Please specify path in config file!"
    post_proc_config.pop('offsets')


    if crop_slice is not None:
        post_proc_config['data_slice'][sample] = crop_slice
    else:
        crop_slice = post_proc_config['data_slice'][sample]

    crop_slice_is_not_none = [slice(None) for _ in range(4)] != parse_data_slice(crop_slice)


    print("Loading affinities and init. segmentation...")
    affinities_from_hdf5 = affinities is None
    init_segm = None
    if given_initSegm:
        if affinities is None:
            affinities, init_segm = import_postproc_data(project_folder, aggl_name=name_aggl,
                             data_to_import=['affinities', 'init_segmentation'],crop_slice=crop_slice)
        else:
            init_segm = import_postproc_data(project_folder, aggl_name=name_aggl,
                                             data_to_import=['init_segmentation'],
                                             crop_slice=crop_slice)

    else:
        if affinities is None:
            affinities= import_postproc_data(project_folder, aggl_name=name_aggl,
                                                         data_to_import=['affinities'],
                                             crop_slice=crop_slice)


    if affinities_from_hdf5:
        affinities_dataset = AffinitiesHDF5VolumeLoader.from_config(post_proc_config['volumes']['affinities'],
                                                                    name=sample, data_slice=crop_slice)
    else:
        if crop_slice_is_not_none:
            slc = tuple(parse_data_slice(crop_slice))
            affinities = affinities[np.s_[slc]]
        affinities_dataset = AffinitiesVolumeLoader.from_config(affinities,
                                                                sample,
                                            post_proc_config['volumes']['affinities'])



    # assert affinities.shape[1:] == init_segm.shape, "{}, {}".format(affinities.shape, init_segm.shape)

    # TODO: improve this
    gt_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    slc = tuple(parse_data_slice(crop_slice))
    # # bb = np.s_[65:71]
    bb = np.s_[slc[1:]]
    with h5py.File(gt_path, 'r') as f:
        gt = f['segmentations/groundtruth_fixed'][bb].astype('uint64')


    if crop_slice_is_not_none:
        print("Crop slice is not None. Running connected components.")
        gt = vigra.analysis.labelVolumeWithBackground(gt.astype('uint32'))
        if init_segm is not None:
            init_segm = vigra.analysis.labelVolume(init_segm.astype('uint32'))


    return_fragments = post_proc_config.pop('return_fragments', False)
    post_proc_config.pop('nb_threads')
    invert_affinities = post_proc_config.pop('invert_affinities', False)
    segm_pipeline_type = post_proc_config.pop('segm_pipeline_type', 'gen_HC')

    segmentation_pipeline = get_segmentation_pipeline(
        segm_pipeline_type,
        offsets,
        nb_threads=n_threads,
        invert_affinities=invert_affinities,
        return_fragments=return_fragments,
        **post_proc_config
    )

    if post_proc_config.get('use_final_agglomerater', False):
        final_agglomerater = FixationAgglomeraterFromSuperpixels(
                        offsets,
                        n_threads=n_threads,
                        invert_affinities=invert_affinities,
                         **post_proc_config['generalized_HC_kwargs']['final_agglomeration_kwargs']
        )
    else:
        final_agglomerater = None


    post_proc_solver = BlockWise(segmentation_pipeline=segmentation_pipeline,
              offsets=offsets,
                                 final_agglomerater=final_agglomerater,
              blockwise=post_proc_config.get('blockwise', False),
              invert_affinities=invert_affinities,
              nb_threads=n_threads,
              return_fragments=return_fragments,
              blockwise_config=post_proc_config.get('blockwise_kwargs', {}))




    print("Starting prediction...")
    tick = time.time()
    if given_initSegm:
        init_segm, _, _ = vigra.analysis.relabelConsecutive(init_segm.astype('uint32'))
        output_segmentations = post_proc_solver(affinities_dataset, init_segm)
    else:
        output_segmentations = post_proc_solver(affinities_dataset)
    pred_segm = output_segmentations[0] if isinstance(output_segmentations, tuple) else output_segmentations
    print("Post-processing took {} s".format(time.time() - tick))
    print("Pred. sahpe: ", pred_segm.shape)
    print("GT shape: ", gt.shape)
    print("Min. GT label: ", gt.min())

    if post_proc_config.get('stacking_2D', False):
        print('2D stacking...')
        stacked_pred_segm = np.empty_like(pred_segm)
        max_label = 0
        for z in range(pred_segm.shape[0]):
            slc = vigra.analysis.labelImage(pred_segm[z].astype(np.uint32))
            stacked_pred_segm[z] = slc + max_label
            max_label += slc.max() + 1
        pred_segm = stacked_pred_segm


    segm_file = os.path.join(postproc_dir, 'pred_segm.h5')
    name_finalSegm = 'finalSegm'
    print("Writing on disk...")
    # TODO: write possible blocks and fragments...
    vigra.writeHDF5(pred_segm.astype('int64'), segm_file, name_finalSegm, compression='gzip')

    # print("Connected components if slice is taken...")
    # gt = vigra.analysis.labelVolumeWithBackground(gt.astype('uint32'))
    # init_segm = vigra.analysis.labelVolumeWithBackground(init_segm.astype('uint32'))
    # pred_segm = vigra.analysis.labelVolumeWithBackground(pred_segm.astype('uint32'))
    # # best_gt = vigra.analysis.labelVolumeWithBackground(best_gt.astype('uint32'))
    # # ignore_mask = best_gt != 0

    # if given_initSegm:
    #     print("Evaluating scores...")
    #     initSegm_evals = cremi_score(gt, init_segm, border_threshold=None, return_all_scores=True)
    #     print("Score of the oversegm:", initSegm_evals)
    print("Computing score...")
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
    # if given_initSegm:
    #     res['init_segm'][sample] = initSegm_evals
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
    parser.add_argument('--use_default_postproc_config', default=False, type=bool)
    parser.add_argument('--model_IDs', nargs='+', default=None, type=str)

    args = parser.parse_args()

    project_directory = args.project_directory

    if project_directory[0] != '/':
        project_directory = os.path.join('/net/hciserver03/storage/abailoni/learnedHC/', project_directory)
    offset_file = args.offset_file
    if offset_file[0] != '/':
        offset_file = os.path.join('/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/', offset_file)


    offsets = parse_offsets(offset_file)
    n_threads = args.n_threads
    name_aggl = args.name_aggl
    name_infer = args.name_infer
    samples = args.samples
    crop_slice = args.crop_slice

    for sample in samples:
        evaluate(project_directory, sample, offsets, n_threads, name_aggl, name_infer,
                 crop_slice, use_default_postproc_config=args.use_default_postproc_config,
                 model_IDs=args.model_IDs)
