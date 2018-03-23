"""Wrapper around HC-constrained-policy implemented in nifty"""

import numpy as np
import nifty.graph.agglo as nagglo

NODE_KEYS = {'node_sizes':0,
             'node_labels': 1,
             'node_GT': 2}

EDGE_KEYS = {'edge_sizes': 0,
             'edge_indicators': 1,
             'dendrogram_heigh': 2,
             'merge_times': 3,
             'loss_targets': 4,
             'loss_weights': 5}


class constrained_hierarchical_clustering(object):
    def __init__(self, rag, edge_sizes, node_sizes, threshold=0.5,
                 GT_labels=None, ignore_label=None,
                 constrained=True, compute_loss_data=True, use_affinities=True, bincount=None,
                 max_nb_milesteps=-1):
        '''
            * edgeIndicators in [0, 1]. We start merging from 1 to 0.

            * returned targets:
                + 1.0 should be merged
                - 1.0 should not be merged

            * ignore_label: label in the passed GT_labels that should be ignored.
                            If a merge involves the ignore_label it is performed, but not included in the training.
                            The resulting merged segment becomes labels as "ignore_label" (the label spreads).

            * linkage update: average (median also implemented in nifty, but not used here)

            * if computeLossData==False, expensive backtracking of edge-union-find is avoided

        '''
        assert bincount is None, "Bincount not used for an average update"
        assert use_affinities, "Now implemented only for affinities"

        if ignore_label is None:
            ignore_label = np.uint64(-1)
        if bincount is None:
            bincount = 256

        self._rag = rag
        self._nb_nodes = nb_nodes = rag.numberOfNodes
        self._nb_edges = nb_edges = rag.numberOfEdges
        self._use_affs = use_affinities
        self._constrained = constrained
        self._compute_loss_data = compute_loss_data
        self._max_nb_milesteps = max_nb_milesteps
        self.isOver = False
        self.iterations_milesteps = []
        self.current_data = {}


        # TODO: deduce node and edge sizes from rag
        # if edge_sizes is None or node_sizes is None:
        #     pass
        # if edge_sizes is None:
        #     edge_sizes = computed_edge_sizes
        # else:
        #     assert edge_sizes.shape == (nb_edges,)
        # if node_sizes is None:
        #     node_sizes = computed_node_sizes
        # else:
        #     assert node_sizes.shape == (nb_nodes,)
        assert node_sizes.shape == (nb_nodes,)
        assert edge_sizes.shape == (nb_edges,)


        if GT_labels is None:
            compute_loss_data = False
            constrained = False
            GT_labels = np.zeros(nb_nodes, dtype=np.int64)
        else:
            assert GT_labels.shape == (nb_nodes,)
            GT_labels = GT_labels.astype(np.int64)

        edge_indicators = np.zeros(rag.numberOfEdges, np.float)

        # if not self._use_affs:
        #     threshold = 1. - threshold

        self._constrained_HC = nagglo.constrainedHierarchicalClusteringWithUcm(
            graph=rag, edgeIndicators=edge_indicators,
            edgeSizes=edge_sizes.astype(np.float), nodeSizes=node_sizes.astype(np.float),
            bincount=bincount, verbose=False, threshold=threshold, # Usual options
            GTlabels=GT_labels, ignore_label=ignore_label,
            constrained=constrained, computeLossData=compute_loss_data)  # Extra options

    def run_next_milestep(self, edge_indicators, nb_iterations=-1):
        """
            * nb_iterations: with -1 runs until threshold is reached
            * return True if the agglomeration is over
        """
        assert edge_indicators.shape == (self._nb_edges,)
        # if not self._use_affs:
        #     edge_indicators = 1 - edge_indicators

        if self.nb_performed_milesteps>=self._max_nb_milesteps-1:
            print("Max number of milesteps reached. Running until termination.")
            nb_iterations = -1

        self.isOver = self._constrained_HC.runNextMilestep(nb_iterations_in_milestep=nb_iterations,
                                                    new_edge_indicators=edge_indicators)

        if self.isOver:
            # Collect nb of iterations:
            self.iterations_milesteps.append(self._constrained_HC.time())
        else:
            self.iterations_milesteps.append(nb_iterations)

        # TODO: do it better... (save data of every iteration?)
        # Collect and store data milestep:
        data_milestep = self._constrained_HC.collectDataMilestep()
        node_sizes, node_labels, node_GT, edge_sizes, new_edge_indicators, dendrogram_heigh, \
        merge_times, loss_targets, loss_weights = data_milestep

        self.current_data['node_sizes'] = node_sizes
        self.current_data['edge_sizes'] = edge_sizes
        self.current_data['node_labels']= node_labels
        self.current_data['node_GT']= node_GT
        self.current_data['edge_indicators'] = new_edge_indicators
        self.current_data['dendrogram_heigh'] = dendrogram_heigh
        self.current_data['merge_times'] = merge_times
        if self._compute_loss_data:
            self.current_data['loss_targets'] = loss_targets
            self.current_data['loss_weights'] = loss_weights


        return self.isOver

    @ property
    def nb_performed_milesteps(self):
        return len(self.iterations_milesteps)

    @property
    def nb_performed_iterations(self):
        if len(self.iterations_milesteps)!=0:
            return self.iterations_milesteps[-1]
        else:
            return 0

    def get_all_last_data_milestep(self):
        """
        Merge together all the node and edge features.

        Returned values (all float for every edge/node of the original rag):

            * node_sizes
            * node_labels
            * node_GT_labels (ignore label is propagated to the merged nodes)

            * edge_sizes            (-1.0. if contracted)
            * edge_indicators       (-1.0  if contracted)
            * dendrogram_heigh      (-1.0  if not contracted)
            * merge_times           (-1.0  if not contracted)
            * loss_targets          (not returned if not computeLossData, + 1.0 should be merged, - 1.0 should not be merged)
            * loss_weights          (not returned if not computeLossData)

        """
        last_data = self.current_data

        node_features = np.stack([last_data['node_sizes'], last_data['node_labels'], last_data['node_GT']], axis=-1)
        edge_features = np.stack([last_data['edge_sizes'], last_data['edge_indicators'], last_data['dendrogram_heigh'], last_data['merge_times']], axis=-1)
        if self._compute_loss_data:
            edge_features_loss = np.stack([last_data['loss_targets'],
                                          last_data['loss_weights']], axis=-1)
            edge_features =  np.concatenate([edge_features, edge_features_loss], axis=-1)

        return node_features, edge_features

