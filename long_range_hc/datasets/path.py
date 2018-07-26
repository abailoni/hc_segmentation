import os
import socket
import getpass
from inferno.utils.io_utils import yaml2dict
import json
import yaml

original_trainvol_path = "/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi"
username = getpass.getuser()


hostname = socket.gethostname()

""""Add your hostname / paths to path.py for hardcoded fun!"""
if hostname == "trendytukan" or username == "abailoni":
    cremi_trainvol_path = "/export/home/abailoni/datasets/cremi/constantin_data"
else:
    cremi_trainvol_path = None




def get_template_config_file(template_path, output_path):
    if cremi_trainvol_path is None:
        # Load the default template_file:
        template = yaml2dict(template_path)
    else:
        # Terrible hack to replace all the cremi paths in the config file:
        original_trainvol_path_mod = original_trainvol_path.replace("/", "\/")
        cremi_trainvol_path_mod = cremi_trainvol_path.replace("/", "\/")
        cmd_string = "sed 's/{}/{}/g' {} > {}".format(original_trainvol_path_mod, cremi_trainvol_path_mod,
                                                      template_path,
                                                      output_path)
        os.system(cmd_string)
        template = yaml2dict(output_path)
    return template


def parse_offsets(offset_file):
    assert os.path.exists(offset_file)
    with open(offset_file, 'r') as f:
        offsets = json.load(f)
    return offsets

def recursive_dict_update(source, target):
    for key, value in source.items():
        if isinstance(value, dict):
            sub_target = target[key] if key in target else {}
            target[key] = recursive_dict_update(source[key], sub_target)
        else:
            target[key] = source[key]
    return target

def adapt_configs_to_model(model_ID,
                            **path_configs):
    for key in path_configs:
        assert key in ['models', 'train', 'valid', 'data', 'postproc']
    assert isinstance(model_ID, int)

    # Load config dicts:
    configs = {}
    for key in path_configs:
        configs[key] = yaml2dict(path_configs[key])

    # Look for the given model:
    model_name = None
    for name in configs['models']:
        if configs['models'][name]['model_ID'] == model_ID:
            model_name = name
            break
    assert model_name is not None, "Model ID not found in the config file"
    print("Using model ", model_name)
    model_configs = configs['models'][model_name]

    # Update model-specific parameters:
    for key in path_configs:
        configs[key] = recursive_dict_update(model_configs.get(key, {}), configs[key])

    # # Update HC threshold postproc:
    # configs['postproc']['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']['threshold'] = \
    #     model_configs['threshold']
    #
    # # Update struct. training options:
    # configs['train']['HC_config']['trained_mistakes'] = model_configs['trained_mistakes']

    # Update paths init. segm and GT:
    samples = ['A', 'B', 'C']
    model_volume_config = model_configs['volumes']

    def update_paths(target_vol_config, source_vol_config):
        # Loop over 'init_segmentation', 'GT', ...
        # If the path is not specified, then the one of 'init_segmentation' will be used
        for input_key in source_vol_config:
            target_vol_config[input_key] = {'dtype': 'int32', 'path': {},
                                                      'path_in_h5_dataset': {}}
            for smpl in samples:
                path = source_vol_config[input_key]['path'] if 'path' in source_vol_config[input_key] else source_vol_config['init_segmentation']['path']
                path = path.replace('$', smpl)
                h5_path = source_vol_config[input_key]['path_in_h5_dataset'].replace('$', smpl)
                target_vol_config[input_key]['path'][smpl] = path
                target_vol_config[input_key]['path_in_h5_dataset'][smpl] = h5_path

        return target_vol_config

    configs['data']['volume_config'] = update_paths(configs['data']['volume_config'], model_volume_config)
    configs['valid']['volume_config'] = update_paths(configs['valid']['volume_config'], model_volume_config)


    # Dump config files to disk:
    for key in path_configs:
        with open(path_configs[key], 'w') as f:
            yaml.dump(configs[key], f)


