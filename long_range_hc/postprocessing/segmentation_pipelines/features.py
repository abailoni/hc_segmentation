import nifty
import numpy as np
from nifty.graph import rag as nrag
import time
from nifty.graph import undirectedLongRangeGridGraph

def accumulate_affinities_on_graph_edges(affinities, offsets, graph=None, label_image=None,
                                         contractedRag=None,
                                         use_undirected_graph=False,
                                         mode="mean",
                                         number_of_threads=6,
                                         offsets_weights=None):
    # TODO: Create class and generalize...
    """
    Label image or graph should be passed. Using nifty rag or undirected graph.

    :param affinities: expected to have the offset dimension as last/first one
    """
    assert mode in ['mean', 'max'], "Only max and mean are implemented"

    if affinities.shape[-1] != offsets.shape[0]:
        assert affinities.shape[0] == offsets.shape[0], "Offsets do not match passed affs"
        ndims = affinities.ndim
        # Move first axis to the last dimension:
        affinities = np.rollaxis(affinities, 0, ndims)

    assert label_image is not None or graph is not None
    if contractedRag is not None:
        assert graph is not None

    if offsets_weights is not None:
        if isinstance(offsets_weights, (list, tuple)):
            offsets_weights = np.array(offsets_weights)
        assert offsets_weights.shape[0] == affinities.shape[-1]
        if all([w>=1.0 for w in offsets_weights]):
            # Take the inverse:
            offsets_weights = 1. / offsets_weights
        else:
            assert all([w<=1.0 for w in offsets_weights]) and all([w>=0.0 for w in offsets_weights])
    else:
        offsets_weights = np.ones(affinities.shape[-1])

    if graph is None:
        graph = nrag.gridRag(label_image.astype(np.uint32))

    if not use_undirected_graph:
        if contractedRag is None:
            raise DeprecationWarning("There is some problem in the nifty function...")
            assert mode == 'mean'
            accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                                              affinities.astype(np.float32),
                                                                              offsets.astype(np.int32),
                                                                              offsets_weights.astype(np.float32),
                                                                              number_of_threads)
        else:
            print("Warning: multipleThread option not implemented!")
            assert mode == 'mean'
            accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph, contractedRag,
                                                                  affinities.astype(np.float32), offsets.astype(np.int32))
    else:
        assert contractedRag is None, "Not implemented atm"
        assert label_image is not None
        # Here 'graph' is actually a general undirected graph (thus label image is needed):
        accumulated_feat, counts, max_affinities = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                                          label_image.astype(np.int32),
                                                                          affinities.astype(np.float32),
                                                                          offsets.astype(np.int32),
                                                                          offsets_weights.astype(np.float32),
                                                                          number_of_threads)
    if mode == 'mean':
        return accumulated_feat, counts
    elif mode == 'max':
        return accumulated_feat, max_affinities

class FeaturerLongRangeAffs(object):
    def __init__(self, offsets,
                       offsets_weights=None,
                       used_offsets=None,
                       debug=True,
                       n_threads=1,
                   invert_affinities=False,
                 statistic='mean',
                 max_distance_lifted_edges=1,
                 return_dict=False):

        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.used_offsets = used_offsets
        self.return_dict = return_dict
        self.offsets_weights = offsets_weights
        self.statistic = statistic


        assert isinstance(n_threads, int)
        assert isinstance(max_distance_lifted_edges, int)

        self.offsets = offsets
        self.debug = debug
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.max_distance_lifted_edges = max_distance_lifted_edges


    def __call__(self, affinities, segmentation):
        tick = time.time()
        offsets = self.offsets
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            offsets_weights = offsets_weights[self.used_offsets]

        assert affinities.ndim == 4
        # affinities = affinities[:3]
        assert affinities.shape[0] == offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        # Build rag and compute node sizes:
        if self.debug:
            print("Computing rag...")
            tick = time.time()
        rag = nrag.gridRag(segmentation.astype(np.uint32))

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            tick = time.time()

        out_dict = {}
        out_dict['rag'] = rag

        if self.max_distance_lifted_edges != 1:
            # Build lifted graph:
            print("Building graph...")
            lifted_graph, is_local_edge = build_lifted_graph_from_rag(
                rag,
                segmentation,
                offsets,
                max_lifted_distance=self.max_distance_lifted_edges,
                number_of_threads=self.n_threads)

            # lifted_graph, is_local_edge, _, edge_sizes = build_pixel_lifted_graph_from_offsets(
            #     segmentation.shape,
            #     offsets,
            #     label_image=segmentation,
            #     offsets_weights=None,
            #     nb_local_offsets=3,
            #     GT_label_image=None
            # )

            if self.debug:
                print("Took {} s!".format(time.time() - tick))
                print("Computing edge_features...")
                tick = time.time()

            # Compute edge sizes and accumulate average/max:
            edge_indicators, edge_sizes = \
                accumulate_affinities_on_graph_edges(
                    affinities, offsets,
                    graph=lifted_graph,
                    label_image=segmentation,
                    use_undirected_graph=True,
                    mode=self.statistic,
                    offsets_weights=offsets_weights,
                    number_of_threads=self.n_threads)
            out_dict['graph'] = lifted_graph
            out_dict['edge_indicators'] = edge_indicators
            out_dict['edge_sizes'] = edge_sizes
        else:
            out_dict['graph'] = rag
            print("Computing edge_features...")
            is_local_edge = np.ones(rag.numberOfEdges, dtype=np.int8)
            # TODO: her we have rag (no need to pass egm.), but fix nifty function first.
            if self.statistic == 'mean':
                edge_indicators, edge_sizes = \
                    accumulate_affinities_on_graph_edges(
                        affinities, offsets,
                        graph=rag,
                        label_image=segmentation,
                        use_undirected_graph=True,
                        mode=self.statistic,
                        offsets_weights=offsets_weights,
                        number_of_threads=self.n_threads)
                out_dict['edge_indicators'] = edge_indicators
                out_dict['edge_sizes'] = edge_sizes
            elif self.statistic == 'max':
                merge_prio, edge_sizes = \
                    accumulate_affinities_on_graph_edges(
                        affinities, offsets,
                        graph=rag,
                        label_image=segmentation,
                        use_undirected_graph=True,
                        mode=self.statistic,
                        offsets_weights=offsets_weights,
                        number_of_threads=self.n_threads)
                not_merge_prio, _ = \
                    accumulate_affinities_on_graph_edges(
                        1 - affinities, offsets,
                        graph=rag,
                        label_image=segmentation,
                        use_undirected_graph=True,
                        mode=self.statistic,
                        offsets_weights=offsets_weights,
                        number_of_threads=self.n_threads)
                edge_indicators = merge_prio
                out_dict['edge_indicators'] = merge_prio
                out_dict['merge_prio'] = merge_prio
                out_dict['not_merge_prio'] = not_merge_prio
                out_dict['edge_sizes'] = edge_sizes


        if not self.return_dict:
            edge_features = np.stack([edge_indicators, edge_sizes, is_local_edge])
            # NOTE: lifted graph is not returned!
            return rag, edge_features
        else:
            out_dict['is_local_edge'] = is_local_edge
            return out_dict


def build_lifted_graph_from_rag(rag,
                                label_image,
                                offsets, max_lifted_distance=3,
                                number_of_threads=6):

    local_edges = rag.uvIds()

    if max_lifted_distance > 1:
        # Search for lifted edges in a certain range (max_dist == 1, then only local)
        long_range_edges = rag.bfsEdges(max_lifted_distance)


        temp_lifted_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        temp_lifted_graph.insertEdges(local_edges)
        nb_local_edges = temp_lifted_graph.numberOfEdges
        temp_lifted_graph.insertEdges(long_range_edges)


        # Check whenever the lifted edges are actually covered by the offsets:
        fake_affs = np.ones(label_image.shape + (offsets.shape[0], ))
        label_image = label_image.astype(np.int32)
        _, edge_sizes = \
            accumulate_affinities_on_graph_edges(fake_affs, offsets,
                                                 graph=temp_lifted_graph,
                                                 label_image=label_image,
                                                 use_undirected_graph=True,
                                                 number_of_threads=number_of_threads)


        # Find lifted edges reached by the offsets:
        edges_to_keep = edge_sizes>0.
        uvIds_temp_graph = temp_lifted_graph.uvIds()

        final_lifted_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        final_lifted_graph.insertEdges(uvIds_temp_graph[edges_to_keep])
        total_nb_edges = final_lifted_graph.numberOfEdges

        is_local_edge = np.zeros(total_nb_edges, dtype=np.int8)
        is_local_edge[:nb_local_edges] = 1

    else:
        final_lifted_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        final_lifted_graph.insertEdges(local_edges)
        total_nb_edges = final_lifted_graph.numberOfEdges
        is_local_edge = np.ones(total_nb_edges, dtype=np.int8)

    return final_lifted_graph, is_local_edge


def build_pixel_lifted_graph_from_offsets(image_shape,
                                          offsets,
                                          label_image=None,
                                          GT_label_image=None,
                                          offsets_probabilities=None,
                                          offsets_weights=None,
                                          nb_local_offsets=3):
    """
    :param offsets: At the moment local offsets should be the first ones
    :param nb_local_offsets: UPDATE AND GENERALIZE!
    """
    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape

    is_local_offset = np.zeros(offsets.shape[0], dtype='bool')
    is_local_offset[:nb_local_offsets] = True
    if label_image is not None:
        assert image_shape == label_image.shape
        if offsets_weights is not None:
            print("Offset weights ignored...!")


    # TODO: change name offsets_probabilities
    print("Actually building graph...")
    tick = time.time()
    graph = undirectedLongRangeGridGraph(image_shape, offsets, is_local_offset,
                        offsets_probabilities=offsets_probabilities,
                        labels=label_image)
    nb_nodes = graph.numberOfNodes
    if label_image is None:
        print("Getting edge index...")
        offset_index = graph.edgeOffsetIndex()
        is_local_edge = offset_index.astype('int32')
        w = np.where(offset_index < nb_local_offsets)
        is_local_edge[:] = 0
        is_local_edge[w] = 1
    else:
        print("Took {} s!".format(time.time() - tick))
        print("Checking edge locality...")
        is_local_edge = graph.findLocalEdges(label_image).astype('int32')


    if offsets_weights is None or label_image is not None:
        edge_weights = np.ones(graph.numberOfEdges, dtype='int32')
    else:
        if isinstance(offsets_weights,(list,tuple)):
            offsets_weights = np.array(offsets_weights)
        assert offsets_weights.shape[0] == offsets.shape[0]

        if all([w>=1.0 for w in offsets_weights]):
            # Take the inverse:
            offsets_weights = 1. / offsets_weights
        else:
            assert all([w<=1.0 for w in offsets_weights]) and all([w>=0.0 for w in offsets_weights])

        print("Edge weights...")
        edge_weights = offsets_weights[offset_index.astype('int32')]


    if GT_label_image is None:
        GT_labels_nodes = np.zeros(nb_nodes, dtype=np.int64)
    else:
        assert GT_label_image.shape == image_shape
        GT_labels_image = GT_label_image.astype(np.uint64)
        GT_labels_nodes = graph.nodeValues(GT_labels_image)

    return graph, is_local_edge, GT_labels_nodes, edge_weights