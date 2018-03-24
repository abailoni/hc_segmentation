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



from long_range_hc.datasets import AffinitiesVolumeLoader
from long_range_hc.postprocessing.long_range_aggl import BlockWise
from long_range_hc.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import \
    FixationAgglomeraterFromSuperpixels

from cremi.evaluation import NeuronIds
from cremi import Volume

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline


def evaluate(project_folder, sample, offsets,
             n_threads, name_aggl):
    pred_path = os.path.join(project_folder,
                             'Predictions',
                             'prediction_sample%s.h5' % sample)

    # TODO: save config files associated to this prediction!
    aff_loader_config = './template_config_HC/post_proc/aff_loader_config.yml'
    aff_loader_config = yaml2dict(aff_loader_config)
    aff_loader_config['path'] = pred_path
    aff_loader_config['sample'] = sample
    aff_loader_config['offsets'] = list(offsets)

    # TODO: it would be really nice to avoid the full loading of the dataset...
    affinities_dataset = AffinitiesVolumeLoader.from_config(aff_loader_config)

    post_proc_config = './template_config_HC/post_proc/post_proc_config.yml'
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

    return_fragments = post_proc_config.get('return_fragments', False)

    segmentation_pipeline = get_segmentation_pipeline(
        post_proc_config.get('segm_pipeline_type', 'gen_HC'),
        offsets,
        nb_threads=n_threads,
        invert_affinities=post_proc_config.get('invert_affinities', False),
        return_fragments=return_fragments,
        MWS_kwargs=post_proc_config.get('MWS_kwargs',{}),
        generalized_HC_kwargs=post_proc_config.get('generalized_HC_kwargs',{})
    )

    final_agglomerater = FixationAgglomeraterFromSuperpixels(
                    offsets,
                    n_threads=n_threads,
                    invert_affinities=post_proc_config.get('invert_affinities', False),
                     **post_proc_config['generalized_HC_kwargs']['final_agglomeration_kwargs']
        #{ 'zero_init': False,
    # 'max_distance_lifted_edges': 5,
    # 'update_rule_merge': 'mean',
    # 'update_rule_not_merge': 'mean'})
    )


    post_proc_solver = BlockWise(segmentation_pipeline=segmentation_pipeline,
              offsets=offsets,
                                 final_agglomerater=final_agglomerater,
              blockwise=post_proc_config.get('blockwise', False),
              invert_affinities=post_proc_config.get('invert_affinities', False),
              nb_threads=post_proc_config.get('nb_threads', False),
              return_fragments=return_fragments,
              blockwise_config=post_proc_config.get('blockwise_kwargs', {}))

    print("Starting prediction...")
    tick = time.time()
    output_segmentations = post_proc_solver(affinities_dataset)
    pred_segm = output_segmentations[0] if isinstance(output_segmentations,tuple) else output_segmentations
    print("Post-processing took {} s".format(time.time() - tick))

    segm_file = os.path.join(postproc_dir, 'pred_segm.h5')
    name_finalSegm = 'finalSegm'
    vigra.writeHDF5(pred_segm.astype('int64'), segm_file, name_finalSegm, compression='gzip')
    if post_proc_config.get('blockwise', True):
        vigra.writeHDF5(output_segmentations[1].astype('int64'), segm_file, name_finalSegm+'_blocks', compression='gzip')
    if return_fragments:
        vigra.writeHDF5(output_segmentations[-1].astype('int64'), segm_file, 'fragments', compression='gzip')

    parsed_slice = parse_data_slice(aff_loader_config['slicing_config']['data_slice'])
    parsed_slice = parsed_slice[1:]


    gt_path = '/export/home/abailoni/datasets/cremi/SOA_affinities/sample%s_train.h5' % (sample)
    # # bb = np.s_[65:71]
    bb = np.s_[tuple(parsed_slice)]
    with h5py.File(gt_path, 'r') as f:
        gt = f['segmentations/groundtruth'][bb].astype('uint64')
    #
    print(gt.shape)
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
    parser.add_argument('--sample', default='B')
    # parser.add_argument('--data_slice', default='85:,:,:')
    parser.add_argument('--n_threads', default=1, type=int)
    parser.add_argument('--name_aggl', default=None)

    args = parser.parse_args()

    project_directory = args.project_directory

    offset_file = args.offset_file
    offsets = parse_offsets(offset_file)
    n_threads = args.n_threads
    name_aggl = args.name_aggl
    sample = args.sample

    evaluate(project_directory, sample, offsets, n_threads, name_aggl)
