import os
import socket
import getpass
from inferno.utils.io_utils import yaml2dict


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