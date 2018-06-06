import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")

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

def predict(project_folder,
            sample,
            offsets,
            data_slice=None,
            only_nn_channels=False,
            ds=None
            ): #Only 3 nearest neighbor channels
    data_config_path = os.path.join(project_folder,
                                    'data_config.yml')
    data_config = yaml2dict(data_config_path)


    gpu = 0
    checkpoint = os.path.join(project_folder, 'Weights')
    if ds == 1:
        infer_config_template_path = './template_config/inference/infer_config.yml'
    elif ds == 2:
        infer_config_template_path = './template_config/inference/infer_config_DS2.yml'
    else:
        raise NotImplementedError()

    infer_config_path = os.path.join(project_folder, 'infer_data_config_sample%s.yml' % sample)
    infer_config = get_template_config_file(infer_config_template_path, infer_config_path)
    # TODO: update config file with dataset name! Update the saved version
    infer_config['dataset_name'] = sample
    infer_config['offsets'] = offsets

    infer_config['volume_config'] = data_config['volume_config']

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
    output = trainer.infer(cremi)

    print("[*] Output has shape {}".format(str(output.shape)))
    save_folder = os.path.join(project_folder, 'Predictions')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, 'prediction_sample%s.h5' % sample)

    # if only_nn_channels:
    #     output = output[:3]
    #     save_path = save_path[:-3] + '_nnaffinities.h5'

    toh5(output.astype('float32'), save_path, compression='lzf')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('project_directory', type=str)
    # parser.add_argument('offset_file', type=str)
    parser.add_argument('--gpus', type=int)
    # parser.add_argument('--data_slice',  default='85:,:,:')
    parser.add_argument('--nb_threads', default=1, type=int)
    # parser.add_argument('--name_aggl', default=None)

    args = parser.parse_args()

    proj_dir = '/net/hciserver03/storage/abailoni/learnedHC/input_segm/'
    offs_dir = '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/'

    offsets_dir = [
        # 'dense_offsets.json',
        # 'dense_offsets.json',
        'SOA_offsets.json'
        # 'SOA_offsets.json'
    ]

    projs = [
        # 'smart_oversegm_DS2_denseOffs',
        # 'smart_oversegm_DS1_denseOffs',
        'WSDT_DS1',
        # 'smart_oversegm_DS2'
    ]

    DS = [
        # 2,
        # 1,
        1,
        # 2
    ]

    for pr_dr, offs, ds  in zip(projs, offsets_dir, DS):


        project_directory = proj_dir + pr_dr
        gpu = args.gpus

        offset_file = offs_dir + offs
        offsets = parse_offsets(offset_file)
        data_slice = None
        n_threads = args.nb_threads
        name_aggl = None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        samples = (
            # 'B',
            'C',
            'A',
        )

        for sample in samples:
            predict(project_directory, sample, offsets, data_slice,
                    only_nn_channels=False,
                    ds=ds)