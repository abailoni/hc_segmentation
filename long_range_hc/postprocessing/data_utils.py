import vigra
from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric.volumetric_utils import parse_data_slice
import numpy as np
import h5py
import os
import json

# TODO: Find a better place
# This at the moment loads segmentations from the usual saved format

def import_postproc_data(proj_dir, aggl_name,
        data_to_import=None,
        crop_slice=None
                    ):
    """
    :param proj_dir:
    :param aggl_name:
    :param data_to_import:
    :param dataset_folder: This should contain files like 'sample%s_train.h5'.
    :param crop_slice: string!
    :return:
    """
    # TODO: generalize file_names in the dataset folder!
    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = config_file['sample']
    volumes = config_file['volumes']

    if data_to_import is None:
        data_to_import = ['raw', 'GT', 'affinities']
    else:
        for data_key in data_to_import:
            if data_key not in volumes:
                raise ValueError('Import key not recognised: {}. Available: {}'.format(data_key, volumes))

    if crop_slice is not None:
        if isinstance(crop_slice, str):
            slc = tuple(parse_data_slice(crop_slice))
        else:
            assert isinstance(crop_slice, tuple)
            slc = crop_slice
        assert len(slc) == 4
    else:
        slc = tuple(parse_data_slice(config_file['data_slice'][sample]))


    bb_affs = np.s_[slc]
    bb = np.s_[slc[1:]]
    outputs = []
    for data_key in data_to_import:
        with h5py.File(volumes[data_key]['path'][sample], 'r') as f:
            bb_vol = bb_affs if data_key == 'affinities' else bb
            dtype = 'float32' if 'dtype' not in volumes[data_key] else volumes[data_key]['dtype']
            vol = f[volumes[data_key]['path_in_h5_dataset'][sample]][bb_vol].astype(dtype)
            if data_key == 'raw':
                vol /= 255.
            outputs.append(vol)
    if len(outputs) == 1:
        return  outputs[0]
    else:
        return tuple(outputs)

def import_SOA_datasets(
        data_to_import=None,
        proj_dir=None,
        aggl_name=None,
        crop_slice=None,
        sample=None,
):
    """
    :param data_to_import: list with ['raw', 'gt', 'affinities']
    :param proj_dir: Only used if we want to load a specific dataset/slice from config
    :param aggl_name: Only used if we want to load a specific dataset/slice from config
    :param dataset_folder: This should contain files like 'sample%s_train.h5'.
    :param crop_slice: This includes the affinity channel!
    :return:
    """
    dataset_folder = '/export/home/abailoni/datasets/cremi/SOA_affinities/'
    # TODO: generalize file_names in the dataset folder!
    if data_to_import is None:
        data_to_import = ['raw', 'gt', 'affinities']
    else:
        for data_key in data_to_import:
            if data_key not in ['raw', 'gt', 'affinities']:
                raise ValueError('Import key not recognised: {}. Available: {}'.format(data_key, ['raw', 'gt', 'affinities']))

    if proj_dir is not None:
        assert aggl_name is not None
        config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
        sample = config_file['sample']
        if 'slicing_config' in config_file:
            # Legacy:
            crop_slice = config_file['slicing_config']['data_slice']
        else:
            crop_slice = config_file['data_slice'][sample]
    else:
        assert sample is not None
        crop_slice = ":,:,:,:" if crop_slice is None else crop_slice


    slc = parse_data_slice(crop_slice)
    dataset_path = os.path.join(dataset_folder,'sample%s_train.h5' % (sample))


    bb_affs = np.s_[tuple(slc)]
    bb = np.s_[tuple(slc[1:])]
    outputs = []
    for data_key in data_to_import:
        if data_key == 'raw':
            with h5py.File(dataset_path, 'r') as f:
                outputs.append(f['raw'][bb].astype(np.float32) / 255.)
        if 'gt' == data_key:
            with h5py.File(dataset_path, 'r') as f:
                outputs.append(f['segmentations/groundtruth_fixed'][bb])
        if 'affinities' == data_key:
            with h5py.File(config_file['path'], 'r') as f:
                outputs.append(f[config_file['path_in_h5_dataset']][bb_affs].astype(np.float32))
    if len(outputs) == 1:
        return  outputs[0]
    else:
        return tuple(outputs)


def import_dataset(proj_dir, aggl_name,
        data_to_import=None,
        dataset_folder = '/export/home/abailoni/datasets/cremi/SOA_affinities/',
        crop_slice=None
                    ):
    """
    SOON DEPRECATED

    :param proj_dir:
    :param aggl_name:
    :param data_to_import:
    :param dataset_folder: This should contain files like 'sample%s_train.h5'.
    :param crop_slice:
    :return:
    """
    # TODO: generalize file_names in the dataset folder!
    if data_to_import is None:
        data_to_import = ['raw', 'gt', 'affinities']
    else:
        for data_key in data_to_import:
            if data_key not in ['raw', 'gt', 'affinities']:
                raise ValueError('Import key not recognised: {}. Available: {}'.format(data_key, ['raw', 'gt', 'affinities']))


    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = config_file['sample']
    dataset_path = os.path.join(dataset_folder,'sample%s_train.h5' % (sample))
    slc = parse_data_slice(config_file['slicing_config']['data_slice'])

    if crop_slice is not None:
        pass



    bb_affs = np.s_[tuple(slc)]
    bb = np.s_[tuple(slc[1:])]
    outputs = []
    for data_key in data_to_import:
        if data_key == 'raw':
            with h5py.File(dataset_path, 'r') as f:
                outputs.append(f['raw'][bb].astype(np.float32) / 255.)
        if 'gt' == data_key:
            with h5py.File(dataset_path, 'r') as f:
                outputs.append(f['segmentations/groundtruth_fixed'][bb])
        if 'affinities' == data_key:
            with h5py.File(config_file['path'], 'r') as f:
                outputs.append(f[config_file['path_in_h5_dataset']][bb_affs].astype(np.float32))
    if len(outputs) == 1:
        return  outputs[0]
    else:
        return tuple(outputs)




def import_segmentations(proj_dir, aggl_name, keys_to_return=None):
    file_path = os.path.join(proj_dir, "postprocess/{}/pred_segm.h5".format(aggl_name))
    if keys_to_return is None:
        keys_to_return = ['finalSegm']
    else:
        with h5py.File(file_path, 'r') as f:
            available_keys = [key for key in f]
            for data_key in keys_to_return:
                if data_key not in available_keys:
                    raise ValueError('Import key not found in file: {}. Availables: {}'.format(data_key, available_keys))


    config_file = yaml2dict(os.path.join(proj_dir, "postprocess/{}/aff_loader_config.yml".format(aggl_name)))
    sample = config_file['sample']

    scores_file = os.path.join(proj_dir, "postprocess/{}/scores.json".format(aggl_name))
    with open(scores_file) as f:
        scores = json.load(f)
    print("{0} --> CS: {1:.4f}; AR: {2:.4f}; VI-split: {3:.4f}; VI-merge: {4:.4f}".format(aggl_name,
                                                                                          scores[sample]['cremi-score'],
                                                                                          scores[sample][
                                                                                              'adapted-rand'],
                                                                                          scores[sample]['vi-split'],
                                                                                          scores[sample]['vi-merge']))

    outputs = []
    for data_key in keys_to_return:
        outputs.append(vigra.readHDF5(file_path, data_key).astype(np.uint32))
    if len(outputs) == 1:
        return  outputs[0]
    else:
        return tuple(outputs)