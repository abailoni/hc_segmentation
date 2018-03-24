# this file provides factories for different postprocessing pipelines

from skunkworks.postprocessing.watershed.wsdt import WatershedOnDistanceTransform, WatershedOnDistanceTransformFromAffinities
from .segmentation_pipelines.agglomeration.fixation_clustering import FixationAgglomerativeClustering
import numpy as np

from skunkworks.postprocessing.watershed import DamWatershed
from skunkworks.postprocessing.watershed.ws import WatershedFromAffinities
from copy import deepcopy

def get_segmentation_pipeline(
        segm_pipeline_type,
        offsets,
        nb_threads=1,
        invert_affinities=False,
        return_fragments=False,
        MWS_kwargs=None,
        generalized_HC_kwargs=None):

    if segm_pipeline_type == 'gen_HC':
        # ------------------------------
        # Build possible fragmenters:
        # ------------------------------
        fragmenter = None
        HC_kwargs = generalized_HC_kwargs
        if HC_kwargs.get('use_fragmenter', False):
            assert 'fragmenter' in HC_kwargs
            fragm_type = HC_kwargs['fragmenter']
            if fragm_type == 'WSDT':
                WSDT_kwargs = deepcopy(HC_kwargs.get('WSDT_kwargs', {}))
                fragmenter = WatershedOnDistanceTransformFromAffinities(
                    offsets,
                    WSDT_kwargs.pop('threshold', 0.5),
                    WSDT_kwargs.pop('sigma_seeds', 0.),
                    invert_affinities=invert_affinities,
                    return_hmap=False,
                    n_threads=nb_threads,
                    **WSDT_kwargs,
                    **HC_kwargs.get('prob_map_kwargs',{}))
            elif fragm_type == 'WS':
                fragmenter = WatershedFromAffinities(
                    offsets,
                    return_hmap=False,
                    invert_affinities=invert_affinities,
                    n_threads=nb_threads,
                    **HC_kwargs.get('prob_map_kwargs', {}))
            else:
                raise NotImplementedError()

        # ------------------------------
        # Build agglomeration:
        # ------------------------------
        nb_local_offsets = HC_kwargs.get('nb_local_offsets', 3)
        prob_LR_edges = HC_kwargs.get('probability_long_range_edges', 1.)
        offsets_probs = np.array([1.] * nb_local_offsets + [prob_LR_edges] * (len(offsets)-nb_local_offsets))

        segm_pipeline = FixationAgglomerativeClustering(
            offsets,
            fragmenter,
            n_threads=nb_threads,
            invert_affinities=invert_affinities,
            return_fragments=return_fragments,
            offsets_probabilities=offsets_probs,
            **HC_kwargs.get('agglomeration_kwargs', {})
        )

    elif segm_pipeline_type == 'MWS':
        segm_pipeline = DamWatershed(offsets,
                                     min_segment_size=10,
                                     invert_affinities=not invert_affinities,
                                   n_threads=nb_threads,
                                   **MWS_kwargs)
    else:
        raise NotImplementedError()
    return segm_pipeline

