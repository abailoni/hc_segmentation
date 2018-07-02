import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/learning/cremi/")

from multiprocessing.pool import ThreadPool
from itertools import repeat

from long_range_hc.trainers.learnedHC.visualization import VisualizationCallback

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

from long_range_hc.datasets.path import get_template_config_file, parse_offsets

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
    # TODO: update config file with dataset name! Update the saved version
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
                 )

    # if only_nn_channels:
    #     output = output[:3]
    #     save_path = save_path[:-3] + '_nnaffinities.h5'
    if dump_affs:
        vigra.writeHDF5(output.astype('float32'), save_path, name_inference, compression='gzip')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('project_directory', type=str)
    # parser.add_argument('offset_file', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    # parser.add_argument('--data_slice',  default='85:,:,:')
    parser.add_argument('--nb_threads', default=1, type=int)
    parser.add_argument('--name_inference', default=None)
    parser.add_argument('--path_init_segm', default=None)
    parser.add_argument('--name_aggl', default=None)
    parser.add_argument('--samples', nargs='+', default=['A', 'B', 'C'], type=str)


    args = parser.parse_args()

    proj_dir = '/net/hciserver03/storage/abailoni/learnedHC/'
    offs_dir = '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/'

    offsets_dir = [
        # 'dense_offsets.json',
        # 'dense_offsets.json',
        'SOA_offsets.json',
        # 'SOA_offsets.json'
    ]

    projs = [
        # 'smart_oversegm_DS2_denseOffs',
        # 'WSDT_DS1_denseOffs',
        'inputSegmPlusBinaryMask/WSDT_DS1',
        # 'look_ahead/WSDT_DS1',
    ]

    DS = [
        # 2,
        # 1,
        1,
        # 1,
        # 2
    ]

    for pr_dr, offs, ds  in zip(projs, offsets_dir, DS):


        project_directory = proj_dir + pr_dr
        gpus = args.gpus

        offset_file = offs_dir + offs
        offsets = parse_offsets(offset_file)
        data_slice = None
        n_threads = args.nb_threads
        name_inference = args.name_inference
        path_init_segm = args.path_init_segm
        name_aggl = args.name_aggl


        # set the proper CUDA_VISIBLE_DEVICES env variables
        gpus = list(args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        gpus = list(range(len(gpus)))

        pool = ThreadPool()
        samples = args.samples


        pool.starmap(predict,
                     zip(samples,
                         gpus,
                         repeat(project_directory),
                         repeat(offsets),
                         repeat(data_slice),
                         repeat(False),
                         repeat(ds),
                         repeat(path_init_segm),
                         repeat(name_inference),
                         repeat(name_aggl)
                         ))

        pool.close()
        pool.join()

