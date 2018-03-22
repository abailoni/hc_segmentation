# this file provides factories for different postprocessing pipelines

from skunkworks.postprocessing.watershed.wsdt import WatershedOnDistanceTransform, WatershedOnDistanceTransformFromAffinities
from .segmentation_pipelines.agglomeration.fixation_clustering import FixationAgglomerativeClustering
from .segmentation_pipelines.agglomeration.fixation_clustering.pipelines_old import FixationAgglomerativeClusteringOld
import numpy as np


# TODO: make this more general
def fixation_agglomerative_clustering_from_wsdt2d(
        offsets,
        threshold=0.4,
        sigma_seeds=2.,
        used_offsets=None,
        offset_weights=None,
        invert_affinities=True,
        min_segment_size=20,
        n_threads=1,
        probability_long_range_edges=0.3,
        return_fragments=False,
        **extra_wsdt_kwargs):

    wsdt = WatershedOnDistanceTransformFromAffinities(
        offsets,
        threshold,
        sigma_seeds,
        used_offsets=used_offsets,
        offset_weights=offset_weights,
        return_hmap=False,
        invert_affinities=invert_affinities,
        min_segment_size=min_segment_size,
        n_threads=n_threads,
    **extra_wsdt_kwargs)

    from skunkworks.postprocessing.watershed.ws import WatershedFromAffinities
    ws = WatershedFromAffinities(
        offsets,
        used_offsets=used_offsets,
        offset_weights=offset_weights,
        return_hmap=False,
        invert_affinities=invert_affinities,
        n_threads=n_threads)

    offsets_probs = np.array([1.] * 3 + [probability_long_range_edges] * 9)


    return FixationAgglomerativeClustering(offsets,
                                           # fragmenter=wsdt,
                                           n_threads=n_threads,
                                           max_distance_lifted_edges=1,
                                           invert_affinities=invert_affinities,
                                           return_fragments=return_fragments,
                                           # update_rule_merge={'name': 'rank', 'q':0.5, 'numberOfBins':200},
                                           # update_rule_not_merge={'name': 'rank', 'q':0.5, 'numberOfBins':200},
                                           update_rule_merge='mean',
                                           update_rule_not_merge='mean',
                                           # update_rule_merge={'name': 'generalized_mean', 'p': 1.0},
                                           # update_rule_not_merge={'name': 'generalized_mean', 'p': 1.0},
                                           zero_init=True,
                                           offsets_probabilities=offsets_probs
                                           )

    # return FixationAgglomerativeClusteringOld(wsdt,
    #                                        offsets,
    #                                        n_threads=n_threads,
    #                                        max_distance_lifted_edges=1,
    #                                        invert_affinities=invert_affinities,
    #                                        return_fragments=return_fragments,
    #                                        zeroInit=False,
    #                                           p0=1.,
    #                                           p1=1.
    #                                        )