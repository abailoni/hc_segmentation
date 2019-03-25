import matplotlib
matplotlib.use('Agg')
import os
# import seaborn
from matplotlib import pyplot as plt
import vigra
from os.path import join
import numpy as np

matplotlib.rcParams.update({'font.size': 5})

plot_folder = '/export/home/abailoni/seminar_talks/talk_october_2018/plots'
seeds_path  = os.path.join(plot_folder, 'plot_random_seeds.h5')

DEF_INTERP = 'none'


import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
from long_range_hc.trainers.learnedHC import visualization as hc_vis

segm_plot_kwargs = {'vmax': 1000000, 'vmin':0}

if os.path.exists(seeds_path):
    seeds = vigra.readHDF5(seeds_path, 'data')
else:
    seeds = np.random.rand(1000000, 3)
    vigra.writeHDF5(seeds, seeds_path, 'data')

rand_cm = matplotlib.colors.ListedColormap(seeds)



import vigra
import os
import json
import numpy as np

from segmfriends.io.load import import_postproc_data, import_SOA_datasets, import_dataset, import_segmentations, \
    parse_offsets

from long_range_hc.datasets.path import get_template_config_file, adapt_configs_to_model
from segmfriends.features.mappings import map_features_to_label_array
from segmfriends.utils.various import cantor_pairing_fct
from segmfriends.features.vigra_feat import accumulate_segment_features_vigra

from multiprocessing.pool import ThreadPool, Pool

import nifty.graph.rag as nrag

from segmfriends.io.save import save_edge_indicators, save_edge_indicators_students

from skunkworks.metrics.cremi_score import cremi_score


def plot_segm(target, segm, z_slice=0, background=None, mask_value=None, highlight_boundaries=True, plot_label_colors=True):
    """Shape of expected background: (z,x,y)"""
    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)

    if mask_value is not None:
        segm = hc_vis.mask_the_mask(segm,value_to_mask=mask_value)
    if plot_label_colors:
        target.matshow(segm[z_slice], cmap=rand_cm, alpha=0.4, interpolation=DEF_INTERP, **segm_plot_kwargs)
    masked_bound = hc_vis.get_masked_boundary_mask(segm)
    if highlight_boundaries:
        target.matshow(masked_bound[z_slice], cmap='gray', alpha=0.6, interpolation=DEF_INTERP)
    return masked_bound



z_context = 5
slice_str = "21:26, 220:720, 200:700"
# crop_slice = parse_data_slice(slice_str)

sub_slice_str = "1:6, 20:520, :500"
# sub_crop_slice = parse_data_slice(sub_slice_str)


plot_folder = '/export/home/abailoni/seminar_talks/talk_october_2018/plots'

SOA_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'
# project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
model_0_whole = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'



offsets = parse_offsets(
    '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/offsets_MWS.json')

def transf_UCM(UCM):
    UCM = UCM.astype('float32')
    nb_nodes = UCM.max()
    boundary_mask = UCM == nb_nodes
    UCM[boundary_mask] = - 1.
    UCM = UCM / UCM.max()
    UCM[boundary_mask] = 1.
    UCM = 1 - UCM
    return UCM, boundary_mask




def plot_affs(img_data, targets, z_slice):
    plot_segm(targets[0, z_slice], img_data['gt'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=True)
    # targets[0, z_slice].matshow(img_data['raw'][z_slice], cmap='gray', interpolation=DEF_INTERP)
    targets[1, z_slice].matshow(img_data['affs'][0][z_slice], cmap='gray', interpolation=DEF_INTERP)
    targets[2, z_slice].matshow(img_data['affs'][7][z_slice], cmap='gray', interpolation=DEF_INTERP)
    targets[3, z_slice].matshow(img_data['affs'][15][z_slice], cmap='gray', interpolation=DEF_INTERP)
    # targets[2, z_slice].matshow(img_data['new_prob_map'][z_slice], cmap='gray', interpolation=DEF_INTERP)


def compare_WS(img_data, targets, z_slice):
    plot_segm(targets[0, z_slice], img_data['gt'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=True)
    targets[1, z_slice].matshow(img_data['affs'][7][z_slice], cmap='gray', interpolation=DEF_INTERP)
    plot_segm(targets[2, z_slice], img_data['max'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)
    plot_segm(targets[3, z_slice], img_data['max_noWS'], z_slice,
                     background=img_data['raw'], highlight_boundaries=False)

def compare_sum_mean(img_data, targets, z_slice):
    plot_segm(targets[0, z_slice], img_data['gt'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=True)
    targets[1, z_slice].matshow(img_data['affs'][7][z_slice], cmap='gray', interpolation=DEF_INTERP)
    plot_segm(targets[2, z_slice], img_data['mean'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)
    plot_segm(targets[3, z_slice], img_data['sum'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)

def compare_mean_max(img_data, targets, z_slice):
    plot_segm(targets[0, z_slice], img_data['gt'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=True)
    targets[1, z_slice].matshow(img_data['affs'][7][z_slice], cmap='gray', interpolation=DEF_INTERP)
    plot_segm(targets[2, z_slice], img_data['mean'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)
    plot_segm(targets[3, z_slice], img_data['max'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)

def compare_all(img_data, targets, z_slice):
    plot_segm(targets[0, z_slice], img_data['gt'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=True)
    # targets[1, z_slice].matshow(img_data['affs'][7][z_slice], cmap='gray', interpolation=DEF_INTERP)
    plot_segm(targets[1, z_slice], img_data['mean'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)
    plot_segm(targets[2, z_slice], img_data['max'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)
    plot_segm(targets[3, z_slice], img_data['sum'], z_slice,
                     background=img_data['raw'], highlight_boundaries=True)


def compare_all_UCM(img_data, targets,z_slice):
    z_slice = 2
    # Segmentations:
    plot_segm(targets[0, 0], img_data['mean'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=False)
    plot_segm(targets[0, 1], img_data['max'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=False)
    plot_segm(targets[0, 2], img_data['sum'], z_slice, mask_value=0,
              background=img_data['raw'], highlight_boundaries=False)
    # UCM:
    targets[1, 0].matshow(img_data['UCM_mean'][z_slice,...,1], cmap='bwr', interpolation=DEF_INTERP)
    targets[1, 1].matshow(img_data['UCM_max'][z_slice, ..., 1], cmap='bwr', interpolation=DEF_INTERP)
    targets[1, 2].matshow(img_data['UCM_sum'][z_slice, ..., 1], cmap='bwr', interpolation=DEF_INTERP)



def make_plot(name, plot_function, nrows=4, ncols=z_context, slice_list=None):
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                           figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    for a in fig.get_axes():
        a.axis('off')

    slice_list = range(z_context) if slice_list is None else slice_list

    for z_slice in slice_list:
        plot_function(segms, ax, z_slice)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.tight_layout()
    if name.endswith('pdf'):
        fig.savefig(join(plot_folder, name), format='pdf')
    else:
        raise NotImplementedError


segms = {}
segms['mean_noWS'], segms['mean'], segms['UCM_mean'] = import_segmentations(model_0_whole, 'inferName_v100k_MEAN_subBlock_B',
                                                   keys_to_return=['finalSegm', 'finalSegm_WS', 'UCM'], crop_slice=sub_slice_str)
segms['sum_noWS'], segms['sum'], segms['UCM_sum'] = import_segmentations(model_0_whole, 'inferName_v100k_GAEC_onlyFewEdges_subBlock_B',
                                                   keys_to_return=['finalSegm', 'finalSegm_WS', 'UCM'], crop_slice=sub_slice_str)
segms['max_noWS'], segms['max'], segms['UCM_max'] = import_segmentations(model_0_whole, 'inferName_v100k_MAX_subBlock_B',
                                                   keys_to_return=['finalSegm', 'finalSegm_WS', 'UCM'], crop_slice=sub_slice_str)
segms['raw'], segms['affs'], segms['gt'] = import_postproc_data(proj_dir=model_0_whole, aggl_name='inferName_v100k_MEAN_subBlock_B', data_to_import=['raw', 'affinities', 'GT'],crop_slice="1:,"+slice_str, )

for key in [ky for ky in segms if 'UCM_' in ky]:
    segms[key], segms[key+"_bound"] = transf_UCM(segms[key])

for key in [ky for ky in segms if '_noWS' in ky]:
    segms[key], _, _ = vigra.analysis.relabelConsecutive(segms[key].astype('uint32'))


for name, fun, kwargs in [ ("affinities.pdf", plot_affs, {}),
                   ("compare_sum_mean.pdf", compare_sum_mean, {}),
                   ("compare_mean_max.pdf", compare_mean_max, {}),
                   ("compare_WS.pdf", compare_WS, {}),
   ]:
    make_plot(name, fun, **kwargs)

make_plot("compare_all_UCM.pdf", compare_all_UCM, nrows=2, ncols=3)
