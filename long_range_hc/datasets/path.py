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

def adapt_configs_to_model(model_IDs,
                           debug=False,
                            **path_configs):
    """
    :param model_ID: can be an int ID or the name of the model
    :param path_configs: list of strings with the paths to .yml files
    """
    for key in path_configs:
        assert key in ['models', 'train', 'valid', 'data', 'postproc', 'infer']

    # Load config dicts:
    configs = {}
    for key in path_configs:
        configs[key] = yaml2dict(path_configs[key])

    def get_model_configs(model_IDs, model_configs=None):
        model_configs = {} if model_configs is None else model_configs
        model_IDs = [model_IDs] if not isinstance(model_IDs, list) else model_IDs

        for model_ID in model_IDs:# Look for the given model:
            # Look for the given model:
            model_name = None
            for name in configs['models']:
                if isinstance(model_ID, int):
                    if 'model_ID' in configs['models'][name]:
                        if configs['models'][name]['model_ID'] == model_ID:
                            model_name = name
                            break
                elif isinstance(model_ID, str):
                    if name == model_ID:
                        model_name = name
                        break
                else:
                    raise ValueError("Model ID should be a int. or a string")
            assert model_name is not None, "Model ID {} not found in the config file".format(model_ID)
            if debug:
                print("Using model ", model_name)


            new_model_configs = configs['models'][model_name]

            # Check parents models and update them recursively:
            if 'parent_model' in new_model_configs:
                model_configs = get_model_configs(new_model_configs['parent_model'], model_configs)

            # Update config with current options:
            model_configs = recursive_dict_update(new_model_configs, model_configs)

        return model_configs

    model_configs = get_model_configs(model_IDs)

    # Update paths init. segm and GT:
    if 'volume_config' in model_configs:
        samples = ['A', 'B', 'C']
        model_volume_config = model_configs['volume_config']

        def update_paths(target_vol_config, source_vol_config):
            # Loop over 'init_segmentation', 'GT', ...
            # If the path is not specified, then the one of 'init_segmentation' will be used
            for input_key in source_vol_config:
                target_vol_config[input_key] = {'dtype': 'int32', 'path': {},
                                                          'path_in_h5_dataset': {}} if input_key not in target_vol_config else target_vol_config[input_key]
                for smpl in samples:
                    path = source_vol_config[input_key]['path'] if 'path' in source_vol_config[input_key] else source_vol_config['init_segmentation']['path']
                    path = path.replace('$', smpl)
                    h5_path = source_vol_config[input_key]['path_in_h5_dataset'].replace('$', smpl)
                    target_vol_config[input_key]['path'][smpl] = path
                    target_vol_config[input_key]['path_in_h5_dataset'][smpl] = h5_path

            return target_vol_config

        for key in ['data', 'valid', 'infer', 'postproc']:
            if key in configs:
                configs[key]['volume_config'] = {} if 'volume_config' not in configs[key] else configs[key][
                    'volume_config']
                configs[key]['volume_config'] = update_paths(configs[key]['volume_config'], model_volume_config)

    # Update model-specific parameters:
    for key in path_configs:
        configs[key] = recursive_dict_update(model_configs.get(key, {}), configs[key])

    # Dump config files to disk:
    for key in path_configs:
        with open(path_configs[key], 'w') as f:
            yaml.dump(configs[key], f)


