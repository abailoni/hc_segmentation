import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/learning/cremi/")

from multiprocessing.pool import ThreadPool
from itertools import repeat

from long_range_hc.trainers.learnedHC.visualization import VisualizationCallback

import torch
import os
import numpy as np
import argparse
import vigra
import yaml

from inferno.utils.io_utils import yaml2dict

from inferno.io.volumetric.volumetric_utils import parse_data_slice
from long_range_hc.datasets.cremi.loaders.cremi_realigned import CREMIDatasetRealigned

from skunkworks.datasets.cremi.loaders import RawVolumeWithDefectAugmentation
from skunkworks.inference import SimpleInferenceEngine
from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import toh5

from long_range_hc.datasets.path import get_template_config_file, parse_offsets, adapt_configs_to_model

from long_range_hc.trainers.learnedHC.trainHC import HierarchicalClusteringTrainer

from post_process import evaluate

# def make_data_config(validation_config_file, offsets, n_batches, max_nb_workers, pretrain, reload_model=False):
#     if not reload_model:
#         template_path = './template_config/validation_config.yml' if not pretrain else './template_config/pretrain/data_config.yml'
#         template = get_template_config_file(template_path, validation_config_file)
#         template['offsets'] = offsets
#     else:
#         # Reload previous settings:
#         template = yaml2dict(validation_config_file)
#     template['loader_config']['batch_size'] = n_batches
#     num_workers = NUM_WORKERS_PER_BATCH * n_batches
#     template['loader_config']['num_workers'] = num_workers if num_workers < max_nb_workers else max_nb_workers
#     with open(validation_config_file, 'w') as f:
#         yaml.dump(template, f)


def predict(sample,
        gpu,
        project_folder,
            offsets,
            data_slice=None,
            only_nn_channels=False,
            ds=None,
            path_init_segm=None,
            name_inference=None,
            name_aggl=None,
            dump_affs=False,
            use_default_postproc_config=False,
            model_IDs=None

            ): #Only 3 nearest neighbor channels
    data_config_path = os.path.join(project_folder,
                                    'data_config.yml')
    data_config = yaml2dict(data_config_path)


    checkpoint = os.path.join(project_folder, 'Weights')
    if ds == 1:
        infer_config_template_path = './template_config/inference/infer_config.yml'
    elif ds == 2:
        infer_config_template_path = './template_config/inference/infer_config_DS2.yml'
    else:
        raise NotImplementedError()

    if name_inference is None:
        name_inference = 'data'


    infer_config_path = os.path.join(project_folder, 'infer_data_config_{}_{}.yml'.format(name_inference, sample))
    infer_config = get_template_config_file(infer_config_template_path, infer_config_path)

    infer_config['dataset_name'] = sample
    infer_config['offsets'] = offsets

    infer_config['volume_config'] = data_config['volume_config']

    if path_init_segm is not None:
        path_init_segm = os.path.join(path_init_segm+sample, 'pred_segm.h5')
        infer_config['volume_config']['init_segmentation']['path'][sample] = path_init_segm

    save_folder = os.path.join(project_folder, 'Predictions')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, 'prediction_sample%s.h5' % sample)

    infer_config['volume_config']['affinities'] = {}
    infer_config['volume_config']['affinities']['path_in_h5_dataset'] = {}
    infer_config['volume_config']['affinities']['path_in_h5_dataset'][sample] = name_inference
    infer_config['volume_config']['affinities']['path'] = {}
    infer_config['volume_config']['affinities']['path'][sample] = save_path

    infer_config['infer_config']['gpu'] = gpu


    # Dump config files:
    with open(infer_config_path, 'w') as f:
        yaml.dump(infer_config, f)


    # Adapt config to the passed model options:
    if model_IDs is not None:
        config_paths = {'models': './template_config/models_config.yml',
                        'infer': infer_config_path}
        adapt_configs_to_model(model_IDs, debug=False, **config_paths)
        infer_config = yaml2dict(infer_config_path)




    # data_config['slicing_config']['data_slice'] = data_slice

    print("[*] Loading CREMI sample {} with configuration at: {}".format(sample,
                                                                         data_config_path))

    # Load CREMI sample

    cremi = CREMIDatasetRealigned.from_config(infer_config, inference_mode=True)

    # Load model
    print("[*] Loading CNN model...")
    trainer = HierarchicalClusteringTrainer(pre_train=True).load(from_directory=checkpoint,
                                                                 best=False).cuda([gpu])
    # model = trainer.model

    infer_config = infer_config['infer_config']
    infer_config['offsets'] = offsets
    trainer.build_infer_engine(infer_config)

    # return None

    output = trainer.infer(cremi)

    print("[*] Output has shape {}".format(str(output.shape)))

    if name_aggl is not None:
        evaluate(project_folder, sample, offsets, n_threads=8,
                 name_aggl=name_aggl,
                 name_infer=name_inference,
                 affinities=output.astype('float32'),
                 use_default_postproc_config=use_default_postproc_config,
                 model_IDs=model_IDs
                 )

    # if only_nn_channels:
    #     output = output[:3]
    #     save_path = save_path[:-3] + '_nnaffinities.h5'
    if dump_affs or name_aggl is None:
        print("Dumping local affinities on disk...")
        vigra.writeHDF5(output.astype('float32'), save_path, name_inference, compression='gzip')
        print("Done!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('project_directory', type=str)
    # parser.add_argument('offset_file', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    # parser.add_argument('--data_slice',  default='85:,:,:')
    parser.add_argument('--nb_threads', default=1, type=int)
    parser.add_argument('--name_inference', default=None)
    parser.add_argument('--projs', nargs='+', default=None, type=str)
    parser.add_argument('--path_init_segm', default=None)
    parser.add_argument('--name_aggl', default=None)
    parser.add_argument('--dump_affs', default=False, type=bool)
    parser.add_argument('--use_default_postproc_config', default=False, type=bool)
    parser.add_argument('--samples', nargs='+', default=['A', 'B', 'C'], type=str)
    parser.add_argument('--postproc_config_version', default=None)
    parser.add_argument('--model_IDs', nargs='+', default=None, type=str)


    args = parser.parse_args()

    proj_dir = '/net/hciserver03/storage/abailoni/learnedHC/'
    offs_dir = '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/'

    offsets_dir = 'offsets_MWS.json'
    # [
    #     # 'dense_offsets.json',
    #     # 'dense_offsets.json',
    #     # 'SOA_offsets.json',
    #     # 'offsets_MWS.json',
    #
    #     # 'offsets_MWS.json'
    # ]

    projs = [
        # 'smart_oversegm_DS2_denseOffs',
        # 'WSDT_DS1_denseOffs',
        # 'plain_unstruct/pureDICE_wholeTrainingSet',
        'model_050_A_v3/pureDICE_wholeDtSet'
        # 'model_090_v2/pureDICE_wholeDtSet'
        # 'model_050_A/pureDICE'
        # 'plain_unstruct/MWSoffs_bound2_addedBCE_001',
    ] if args.projs is None else args.projs

    DS = 1
    #     [
    #     # 2,
    #     # 1,
    #     # 1,
    #     1,
    #     # 1,
    #     # 2
    # ]

    # set the proper CUDA_VISIBLE_DEVICES env variables
    gpus = list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    for pr_dr, offs, ds  in zip(projs, repeat(offsets_dir), repeat(DS)):


        project_directory = proj_dir + pr_dr

        offset_file = offs_dir + offs
        offsets = parse_offsets(offset_file)
        data_slice = None
        n_threads = args.nb_threads
        name_inference = args.name_inference
        path_init_segm = args.path_init_segm
        name_aggl = args.name_aggl



        samples = args.samples

        for sample in samples:
            predict(sample,
                    gpus[0],
                    project_directory,
                    offsets,
                    data_slice,
                    False,
                    ds,
                    path_init_segm,
                    name_inference,
                    name_aggl,
                    args.dump_affs,
                    args.use_default_postproc_config,
                    model_IDs=args.model_IDs)
            torch.cuda.empty_cache()

        # # pool = ThreadPool()
        #
        #     pool.starmap(predict,
        #              zip(samples,
        #                  gpus,
        #                  repeat(project_directory),
        #                  repeat(offsets),
        #                  repeat(data_slice),
        #                  repeat(False),
        #                  repeat(ds),
        #                  repeat(path_init_segm),
        #                  repeat(name_inference),
        #                  repeat(name_aggl)
        #                  ))
        # #
        # # pool.close()
        # # pool.join()

