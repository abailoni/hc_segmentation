# FIXME:
import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

from train_with_offset_HC import MultiScaleLossMaxPool, parse_offsets
import os
import numpy as np
import argparse
import vigra
import h5py
import json
import time
import yaml

from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric.volumetric_utils import parse_data_slice



from long_range_hc.datasets import AffinitiesHDF5VolumeLoader
from segmfriends.algorithms.blockwise import BlockWise
from long_range_hc.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import \
    FixationAgglomeraterFromSuperpixels

from cremi.evaluation import NeuronIds
from cremi import Volume

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline


def evaluate(project_folder, sample, offsets,
             n_threads, name_aggl, name_infer=None):
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

    post_proc_config = './template_config_HC/post_proc/post_proc_config.yml'
    post_proc_config = yaml2dict(post_proc_config)

    data_config = yaml2dict(data_config_path)

    # TODO: save config files associated to this prediction!
    aff_loader_config = './template_config_HC/post_proc/aff_loader_config.yml'
    aff_loader_config = yaml2dict(aff_loader_config)
    # aff_loader_config['volumes']['affinities']['path'] = {sample: pred_path}
    # aff_loader_config['volumes']['affinities']['path_in_h5_dataset'] = {sample: aff_path_in_h5file}
    # given_initSegm = post_proc_config['start_from_given_segm']
    # aff_loader_config = yaml2dict(aff_loader_config)
    aff_loader_config['path'] = pred_path
    aff_loader_config['path_in_h5_dataset'] = aff_path_in_h5file
    aff_loader_config['sample'] = sample
    aff_loader_config['offsets'] = list(offsets)



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

    aff_loader_config.pop('data_slice_not_padded', None)
    parsed_slice = parse_data_slice(aff_loader_config['slicing_config']['data_slice'])

    assert not post_proc_config.get('start_from_given_segm', False)



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

    # TODO: it would be really nice to avoid the full loading of the dataset...
    print("Loading affinities and init. segmentation...")
    affinities_dataset = AffinitiesHDF5VolumeLoader.from_config(aff_loader_config)
    print(affinities_dataset.base_sequence)

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
    output_segmentations = post_proc_solver(affinities_dataset)
    pred_segm = output_segmentations[0] if isinstance(output_segmentations,tuple) else output_segmentations
    print("Post-processing took {} s".format(time.time() - tick))
    print(pred_segm.shape)

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
    vigra.writeHDF5(pred_segm.astype('int64'), segm_file, name_finalSegm, compression='gzip')
    if post_proc_config.get('blockwise', True):
        vigra.writeHDF5(output_segmentations[1].astype('int64'), segm_file, name_finalSegm+'_blocks', compression='gzip')
    if return_fragments:
        vigra.writeHDF5(output_segmentations[-1].astype('int64'), segm_file, 'fragments', compression='gzip')


    parsed_slice = parsed_slice[1:]


    gt_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    # # bb = np.s_[65:71]
    bb = np.s_[tuple(parsed_slice)]
    with h5py.File(gt_path, 'r') as f:
        gt = f['segmentations/groundtruth_fixed'][bb].astype('uint64')
    #
    print(gt.shape)
    print(gt.min())
    #
    #


    evals = cremi_score(gt, pred_segm, border_threshold=None, return_all_scores=True)
    print(evals)

    eval_file = os.path.join(postproc_dir, 'scores.json')
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
    parser.add_argument('--samples', nargs='+', default=['A', 'B', 'C'], type=str)
    # parser.add_argument('--data_slice', default='85:,:,:')
    parser.add_argument('--n_threads', default=1, type=int)
    parser.add_argument('--name_aggl', default=None)
    parser.add_argument('--name_infer', default=None)

    args = parser.parse_args()

    project_directory = args.project_directory

    offset_file = args.offset_file
    offsets = parse_offsets(offset_file)
    n_threads = args.n_threads
    name_aggl = args.name_aggl
    samples = args.samples

    for sample in samples:
        evaluate(project_directory, sample, offsets, n_threads, name_aggl, name_infer=args.name_infer)
