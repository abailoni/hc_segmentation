import sys

import segmfriends.transform.segm_to_bound

sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np

from segmfriends.io.load import import_dataset, import_segmentations

project_folder = '/export/home/abailoni/learnedHC/new_experiments/SOA_affinities'

import nifty.graph.rag as nrag

from segmfriends.utils.graph import build_lifted_graph_from_rag


def get_boundary_IDs(aggl_name):
    print("Loading segm {}...".format(aggl_name))
    key = 'finalSegm_WS'
    if 'fullB' in aggl_name:
        key += "_full"
    segm = import_segmentations(project_folder, aggl_name,
                                     keys_to_return=[key])

    segm = segm.astype(np.uint32)

    offsets = np.array([[-1,0,0], [0, -5, 0], [0, 0, -5]])

    print("Building graph...")
    rag = nrag.gridRag(segm)
    # Build lifted graph:
    lifted_graph, _ = build_lifted_graph_from_rag(
        rag,
        segm,
        offsets,
        max_lifted_distance=1,
        number_of_threads=8)

    print("Computing boundary IDs...")
    bound_IDs = segmfriends.transform.segm_to_bound.compute_mask_boundaries_graph(
        offsets,
        graph=lifted_graph,
        label_image=segm,
        return_boundary_IDs=True,
        channel_axis=0,
        use_undirected_graph=True,
        number_of_threads=8
    )
    # Inner parts of the segments receive label -1, change it to 0:
    bound_IDs += 1

    print("Saving...")
    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(bound_IDs, file_path, 'boundary_IDs', compression='gzip')


for aggl_name in [
        'fancyOverseg_betterWeights_fullA_thresh093_blckws',
                  'fancyOverseg_betterWeights_fullC_thresh093_blckws',
                  'fancyOverseg_betterWeights_fullB_thresh093_blckws_1',
    # 'fancyOverseg_szRg00_LREbetterWeights_fullB_thresh093_blckws_2',
]:
    get_boundary_IDs(aggl_name)
