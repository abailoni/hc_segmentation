from nifty.graph import rag as nrag
import nifty.graph.agglo as nagglo
import numpy as np

# TODO: check if everything is in nifty plmc branch

from skunkworks.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering.utils import \
    build_pixel_lifted_graph_from_offsets, build_lifted_graph_from_rag
from skunkworks.postprocessing.segmentation_pipelines.base import SegmentationPipeline
from ...features import accumulate_affinities_on_graph_edges
from skunkworks.criteria.learned_HC.utils import segm_utils as segm_utils



class FixationAgglomerativeClustering(SegmentationPipeline):
    def __init__(self, offsets, fragmenter=None,
                 update_rule_merge='mean', update_rule_not_merge='mean',
                 zero_init=False,
                 max_distance_lifted_edges=3,
                 offsets_probabilities=None,
                 n_threads=1,
                 invert_affinities=False,
                 **super_kwargs):
        """
        If a fragmenter is passed (DTWS, SLIC, etc...) then the agglomeration is done
        starting from superpixels.

        Alternatively, agglomeration starts from pixels.

        Remarks:
          - the initial SP accumulation at the moment is always given
            by an average!
          - it expects REAL affinities (1.0 = merge, 0. = not merge).
            If the opposite is passed, use `invert_affinities`
        """
        if fragmenter is not None:
            agglomerater = FixationAgglomeraterFromSuperpixels(
                offsets,
                max_distance_lifted_edges=max_distance_lifted_edges,
                update_rule_merge=update_rule_merge,
                update_rule_not_merge=update_rule_not_merge,
                zero_init=zero_init,
                n_threads=n_threads,
                invert_affinities=invert_affinities)
            super(FixationAgglomerativeClustering, self).__init__(fragmenter, agglomerater, **super_kwargs)
        else:
            agglomerater = FixationAgglomerater(
                offsets,
                update_rule_merge=update_rule_merge,
                update_rule_not_merge=update_rule_not_merge,
                zero_init=zero_init,
                n_threads=n_threads,
                offsets_probabilities=offsets_probabilities,
                invert_affinities=invert_affinities)
            super(FixationAgglomerativeClustering, self).__init__(agglomerater, **super_kwargs)



class FixationAgglomeraterBase(object):
    def __init__(self, offsets,
                 update_rule_merge='mean', update_rule_not_merge='mean',
                 zero_init=False,
                 n_threads=1,
                 invert_affinities=False):
        """
                Starts from pixels.

                Examples of accepted update rules:

                 - 'mean'
                 - 'max'
                 - 'min'
                 - {name: 'rank', q=0.5, numberOfBins=40}
                 - {name: 'generalized_mean', p=2.0}   # 1.0 is mean
                 - {name: 'smooth_max', p=2.0}   # 0.0 is mean

                """
        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)


        self.passed_rules = [update_rule_merge, update_rule_not_merge]
        self.update_rules = [self.parse_update_rule(rule) for rule in self.passed_rules]

        assert isinstance(zero_init, bool)
        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.zeroInit = zero_init
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities

    def parse_update_rule(self, rule):
        accepted_rules_1 = ['max', 'min', 'mean', 'ArithmeticMean']
        accepted_rules_2 = ['generalized_mean', 'rank', 'smooth_max']
        if not isinstance(rule, str):
            rule = rule.copy()
            assert isinstance(rule, dict)
            rule_name = rule.pop('name')
            p = rule.get('p')
            q = rule.get('q')
            assert rule_name in accepted_rules_1 + accepted_rules_2
            assert not (p is None and q is None)
            parsed_rule = nagglo.updatRule(rule_name, **rule)
        else:
            assert rule in accepted_rules_1
            parsed_rule = nagglo.updatRule(rule)

        return parsed_rule

    def __getstate__(self):
        state_dict = dict(self.__dict__)
        state_dict.pop('update_rules', None)
        return state_dict

    def __setstate__(self, state_dict):
        # if 'passed_rules' in state_dict:
        #     state_dict['update_rules'] = [self.parse_update_rule(rule) for rule in state_dict['passed_rules']]
        self.__dict__.update(state_dict)




class FixationAgglomeraterFromSuperpixels(FixationAgglomeraterBase):
    def __init__(self, *super_args, max_distance_lifted_edges=3,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(FixationAgglomeraterFromSuperpixels, self).__init__(*super_args, **super_kwargs)

        assert isinstance(max_distance_lifted_edges, int)
        self.max_distance_lifted_edges = max_distance_lifted_edges

    def __call__(self, affinities, segmentation):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        assert affinities.ndim == 4
        # affinities = affinities[:3]
        assert affinities.shape[0] == self.offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        # Build rag and compute node sizes:
        rag = nrag.gridRag(segmentation.astype(np.uint32))

        # Build lifted graph:
        lifted_graph, is_local_edge = build_lifted_graph_from_rag(
            rag,
            segmentation,
            self.offsets,
            max_lifted_distance=self.max_distance_lifted_edges,
            number_of_threads=self.n_threads)


        # Compute edge sizes and accumulate average/max:
        edge_indicators, edge_sizes = \
            accumulate_affinities_on_graph_edges(
                affinities, self.offsets,
                graph=lifted_graph,
                label_image=segmentation,
                use_undirected_graph=True,
                mode='mean',
                number_of_threads=self.n_threads)

        merge_prio = edge_indicators
        not_merge_prio = 1. - edge_indicators

        node_sizes = np.squeeze(segm_utils.accumulate_segment_features_vigra(segmentation,
                                                                  segmentation, statistics=["Count"],
                                                                  normalization_mode=None, map_to_image=False))

        cluster_policy = nagglo.fixationClusterPolicy(graph=lifted_graph,
                                                      mergePrios=merge_prio,
                                                      notMergePrios=not_merge_prio,
                                                      edgeSizes=edge_sizes,
                                                      nodeSizes=node_sizes,
                                                      isMergeEdge=is_local_edge,
                                                      updateRule0=self.update_rules[0],
                                                      updateRule1=self.update_rules[1],
                                                      zeroInit=self.zeroInit,
                                                      sizeRegularizer=0.)

        # Run agglomerative clustering:
        agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)
        agglomerativeClustering.run() # (True, 10000)
        node_labels = agglomerativeClustering.result()

        final_segm = segm_utils.map_features_to_label_array(
            segmentation,
            np.expand_dims(node_labels, axis=-1),
            number_of_threads=self.n_threads
        )[..., 0]

        return final_segm






class FixationAgglomerater(FixationAgglomeraterBase):
    def __init__(self, *super_args, offsets_probabilities=None,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(FixationAgglomerater, self).__init__(*super_args, **super_kwargs)

        self.offsets_probabilities = offsets_probabilities


    def __call__(self, affinities):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        assert affinities.ndim == 4
        assert affinities.shape[0] == self.offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        image_shape = affinities.shape[1:]

        # Build graph:
        graph, is_local_edge, _, edge_sizes = \
            build_pixel_lifted_graph_from_offsets(
                image_shape,
                self.offsets,
                offsets_probabilities=self.offsets_probabilities,
                nb_local_offsets=3
            )
        print("Number of edges in graph", graph.numberOfEdges)
        print("Number of nodes in graph", graph.numberOfNodes)

        # Build policy:
        print(edge_sizes)
        # edge_sizes = np.ones(graph.numberOfEdges, dtype='float32')
        node_sizes = np.ones(graph.numberOfNodes, dtype='float32')
        merge_prio = graph.edgeValues(np.rollaxis(affinities,0,4))
        not_merge_prio = 1. - merge_prio
        cluster_policy = nagglo.fixationClusterPolicy(graph=graph,
                              mergePrios=merge_prio, notMergePrios=not_merge_prio,
                              edgeSizes=edge_sizes, nodeSizes=node_sizes,
                              isMergeEdge=is_local_edge,
                              updateRule0=self.update_rules[0],
                              updateRule1=self.update_rules[1],
                              zeroInit=self.zeroInit,
                              sizeRegularizer=0.5)
                              # sizeThreshMin=10.,
                              # sizeThresMax=30.)


        # Run agglomerative clustering:
        agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)
        agglomerativeClustering.run(verbose=True, printNth=500000)
        nodeSeg = agglomerativeClustering.result()

        segmentation = nodeSeg.reshape(image_shape)

        return segmentation



