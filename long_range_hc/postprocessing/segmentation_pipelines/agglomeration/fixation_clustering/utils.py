import nifty
import numpy as np

from skunkworks.postprocessing.segmentation_pipelines.features import accumulate_affinities_on_graph_edges


def build_pixel_lifted_graph_from_offsets(image_shape,
                                          offsets,
                                          GT_label_image=None,
                                          offsets_probabilities=None,
                                          nb_local_offsets=3):
    """
    :param offsets: At the moment local offsets should be the first ones
    :param nb_local_offsets:
    """
    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape

    graph = nifty.graph.undirectedLongRangeGridGraph(shape=image_shape, offsets=offsets, offsets_probabilities=offsets_probabilities)
    offset_index = graph.edgeOffsetIndex()
    nb_nodes = graph.numberOfNodes

    is_local_edge = offset_index.astype('int32')
    w = np.where(offset_index < nb_local_offsets)
    is_local_edge[:] = 0
    is_local_edge[w] = 1

    offsets_weights = np.array(
        [ 1., 1., 1.,
          2., 3, 3, 3, 9, 9, 4, 27, 27
        ]
    )
    offsets_weights = 1. / offsets_weights
    edge_weights = offsets_weights[offset_index.astype('int32')]


    if GT_label_image is None:
        GT_labels_nodes = np.zeros(nb_nodes, dtype=np.int64)
    else:
        assert GT_label_image.shape == image_shape
        GT_labels_image = GT_label_image.astype(np.uint64)
        GT_labels_nodes = graph.nodeValues(GT_labels_image)

    return graph, is_local_edge, GT_labels_nodes, edge_weights


def build_lifted_graph_from_rag(rag,
                                label_image,
                                offsets, max_lifted_distance=3,
                                number_of_threads=6):

    local_edges = rag.uvIds()
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

    return final_lifted_graph, is_local_edge