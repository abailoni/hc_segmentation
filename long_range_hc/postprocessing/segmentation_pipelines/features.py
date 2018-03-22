import numpy as np
from nifty.graph import rag as nrag

def accumulate_affinities_on_graph_edges(affinities, offsets, graph=None, label_image=None,
                                         contractedRag=None,
                                         use_undirected_graph=False,
                                         mode="mean",
                                         number_of_threads=6):
    # TODO: Create class and generalize...
    """
    Label image or graph should be passed. Using nifty rag or undirected graph.

    :param affinities: expected to have the offset dimension as last/first one
    """
    if mode != "mean":
        raise NotImplementedError

    if affinities.shape[-1] != offsets.shape[0]:
        assert affinities.shape[0] == offsets.shape[0], "Offsets do not match passed affs"
        ndims = affinities.ndim
        # Move first axis to the last dimension:
        affinities = np.rollaxis(affinities, 0, ndims)

    assert label_image is not None or graph is not None
    if contractedRag is not None:
        assert graph is not None

    if graph is None:
        graph = nrag.gridRag(label_image.astype(np.uint32))

    if not use_undirected_graph:
        if contractedRag is None:
            accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                                  affinities.astype(np.float32),
                                                                  offsets.astype(np.int32),
                                                                  number_of_threads)
        else:
            print("Warning: multipleThread option not implemented!")
            accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph, contractedRag,
                                                                  affinities.astype(np.float32), offsets.astype(np.int32))
    else:
        assert contractedRag is None, "Not implemented atm"
        assert label_image is not None
        # Here 'graph' is actually a general undirected graph (thus label image is needed):
        accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                              label_image.astype(np.int32),
                                                              affinities.astype(np.float32),
                                                              offsets.astype(np.int32),
                                                              number_of_threads)

    return accumulated_feat, counts