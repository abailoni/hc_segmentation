"""
This module implements the learned Watershed algorithm as a pytorch module
"""
import time

import torch.nn as nn
from torch import from_numpy
from torch.autograd import Variable
from torch import stack
import numpy as np
from copy import deepcopy

import vigra
# FIXME imports!!
from ...postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import constrained_fixation_policy
from ...postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import utils
from ...postprocessing.segmentation_pipelines import features

import skunkworks.postprocessing.util
from skunkworks.metrics.cremi_score import cremi_score

import nifty.graph.rag as nrag

from .utils import segm_utils
from .utils import rag_utils
from . import constrained_hierarchical_clustering as constr_HC

from skunkworks.postprocessing.watershed.wsdt import WatershedOnDistanceTransformFromAffinities
from skunkworks.postprocessing.watershed.ws import WatershedFromAffinities

from .utils.segm_utils_CY import find_best_agglomeration

from inferno.utils.io_utils import yaml2dict

eKeys = constr_HC.EDGE_KEYS
nKeys = constr_HC.NODE_KEYS


class LHC_Worker(nn.Module):
    def __init__(self, options):
        raise DeprecationWarning()
        super(LHC_Worker, self).__init__()

        options = yaml2dict(options)
        self.options = options

        config_HC = options.get('HC_config')

        self._training_data_available = config_HC['training_data_available']

        # Model options:
        self._offsets = np.array(config_HC['offsets'], dtype=np.int)
        self._nb_offsets = self._offsets.shape[0]
        self._input_channels = config_HC.get('input_channels')
        self._dims_prediction_model = dim_pred = config_HC.get('dims_prediction_model', 2)
        assert dim_pred==2 or dim_pred==3
        self._nb_threads = config_HC['nb_threads']


        # Data options:
        self._ignore_label_GT_volume = config_HC.get('ignore_label', -1)

        # LHC options:
        self._HC_window_size_train = tuple(config_HC['HC_window_size_train'])
        self._HC_window_size_valid = tuple(config_HC['HC_window_size_valid'])
        self._HC_window_size = None
        self._max_nb_milesteps = config_HC.get('max_nb_milesteps', -1)
        self._fixation_kwargs = config_HC.get('fixation_kwargs', {})
        self._weight_inner_label = self._fixation_kwargs.pop('weight_inner_label', 1.0)


        self._init_segm_opts = config_HC.get('init_segm', {})
        if self._init_segm_opts.get('use_wsdt', False):
            wsdt_kwargs = self._init_segm_opts.get('wsdt_kwargs', {})
            hmap_kwargs = self._init_segm_opts.get('prob_map_kwargs', {})
            self._wsdt = WatershedOnDistanceTransformFromAffinities(
                self._offsets,
                n_threads=self._nb_threads,
                invert_affinities=False,
                return_hmap=True,
                **wsdt_kwargs,
                **hmap_kwargs)
        elif self._init_segm_opts.get('use_ws', False):
            hmap_kwargs = self._init_segm_opts.get('prob_map_kwargs', {})
            self._ws = WatershedFromAffinities(self._offsets,
                                               n_threads=self._nb_threads,
                                               stacked_2d=True,
                                               invert_affinities=False,
                                               return_hmap=True,
                                               **hmap_kwargs)

        # TODO: understand exactly how it is used:
        self._validation_mode = None

        # Initialize attributes:
        self.clear()

    def forward(self, prediction_gpu):
        """
        :param prediction_gpu: (offsets, 3, z, x, y)
        """
        prediction_all_classes = prediction_gpu.cpu().data.numpy()

        # Three classes predictions: (merge_prob, split_prob, inner_prob)
        prediction_merge_prio = prediction_all_classes[:,0]
        prediction_notMerge_prio = prediction_all_classes[:,1]
        prediction_inner = prediction_all_classes[:,2]

        # TODO: temp modification, not sure it works. Re-normalize classes:
        # prediction_merge_prio = prediction_merge_prio / ( prediction_merge_prio + prediction_notMerge_prio + 0.001)

        self.add_cache_data('prediction_merge_prio', prediction_merge_prio, 'current')
        self.add_cache_data('prediction_notMerge_prio', prediction_notMerge_prio, 'current')
        self.add_cache_data('prediction_inner', prediction_inner, 'current')



        # TODO: add possible noise, etc...

        self.perform_HC_milestep(prediction_merge_prio)

        loss_targets, loss_weights = self.collect_targets_and_weights_milestep()

        if self._dims_prediction_model == 2:
            # Get rid of z axis:
            loss_targets = loss_targets[:,0]
            loss_weights = loss_weights[:,0]

        # Convert back to Variable:
        loss_targets = Variable(from_numpy(loss_targets))
        loss_weights = Variable(from_numpy(loss_weights))
        if prediction_gpu.is_cuda:
            loss_targets = loss_targets.cuda()
            loss_weights = loss_weights.cuda()

        return loss_targets, loss_weights

    def perform_HC_milestep(self, prediction_merge_prio):
        """
        :param prediction: (channels,z,x,y) or (channels,x,y)
        """
        # assert self._iterations_in_next_milestep is not None, "Lookahead has not been computed"



        # tick = time.time()


        edge_indicators = self.accumulate_affinities(prediction_merge_prio)
        # print("Accumulate affs: ", time.time()-tick)

        # tick = time.time()

        if self._max_nb_milesteps > 0:
            self._iterations_in_next_milestep = self.compute_lookahead(edge_indicators)
        else:
            self._iterations_in_next_milestep = -1
        self._is_finished = self._constrained_HC.run_next_milestep(
                                                edge_indicators,
                                                self._iterations_in_next_milestep)
        if self._is_finished and self._max_nb_milesteps>0:
            iterations = np.array(self._constrained_HC.iterations_milesteps)
            iter_mod = np.append(iterations, 0)
            iter_mod = (iter_mod-np.roll(iter_mod, 1))[:-1]
            print("Iter. in each milesteps: {} (Tot: {} in {} milesteps)".format(
                list(iter_mod),
                self._constrained_HC.nb_performed_iterations,
                self._current_milestep+1))
        # print("Milestep HC + lookahead:", time.time()-tick)

        node_features, edge_features = self._constrained_HC.get_all_last_data_milestep()

        # Get rid of some data (do not map them to image):
        used_eKeys = deepcopy(eKeys)
        used_nKeys = deepcopy(nKeys)

        compute_loss_data = self._training_data_available and not self.validation_mode
        if compute_loss_data:
            # Change targets to:
            #   +1.0 --> should merge
            #   0.0  --> should split
            # TODO: fix hack --> currently 'loss_targets' are merge-targets and 'loss_weights' are split-targets
            max_index = int(np.array([value for _, value in used_eKeys.items()]).max())
            used_eKeys['merge_targets'] = max_index + 1
            used_eKeys['split_targets'] = max_index + 2
            used_eKeys.pop('loss_targets')
            new_edge_feat = np.zeros((edge_features.shape[0], 2))

            split_targs = edge_features[...,eKeys['loss_targets']]==-1.0
            new_edge_feat[...,0] = edge_features[...,eKeys['loss_targets']]
            new_edge_feat[...,0][split_targs] = 0.0
            new_edge_feat[...,1] = split_targs.astype(np.float32)

            edge_features = np.concatenate([edge_features, new_edge_feat], axis=-1)


        # Cache computed data:
        if not compute_loss_data:
            used_eKeys.pop('loss_targets')
            used_eKeys.pop('loss_weights')

        # used_eKeys.pop('edge_sizes')
        used_eKeys.pop('merge_times')
        # used_nKeys.pop('node_sizes')
        # used_nKeys.pop('node_GT')

        # Map collected data back to image:
        map_node_feat = segm_utils.map_features_to_label_array(
            self._init_segm,
            node_features[..., [id for _, id in used_nKeys.items()]],
            number_of_threads=self._nb_threads
        )

        # map_edge_feat = segm_utils.map_features_to_label_array(
        #     self._init_boundMask_IDs,
        #     edge_features[..., [id for _, id in used_eKeys.items()]],
        #     ignore_label=-1,
        #     number_of_threads=self._nb_threads
        # )


        mappings_fill_values = {'edge_indicators': -1.,
                            'merge_targets': 0.,
                            'split_targets': 0.,
                            'loss_weights': 0.,
                            'dendrogram_heigh': 0.,
                             'edge_sizes':0.}

        for id, edge_feat in enumerate(used_eKeys):
            mapped_feat = segm_utils.map_features_to_label_array(
                self._init_boundMask_IDs,
                edge_features[..., [used_eKeys[edge_feat]]],
                ignore_label=-1,
                fill_value=mappings_fill_values[edge_feat],
                number_of_threads=self._nb_threads
            )
            self.add_cache_data(['cHC_data/vect/'+edge_feat, 'cHC_data/img/'+edge_feat],
                                [edge_features[...,used_eKeys[edge_feat]], mapped_feat[...,0]],
                                milestep='current')


        for id, node_feat in enumerate(used_nKeys):
            self.add_cache_data(['cHC_data/vect/'+node_feat, 'cHC_data/img/'+node_feat],
                                [node_features[...,used_nKeys[node_feat]], map_node_feat[...,id]],
                                milestep='current')




    def get_dynamic_inputs_milestep(self):
        """
        It also performs lookahead based on previous predictions and decide how many
        iterations should be done in the next milestep.

        :return: dictionary and key_list necessry to compose the CNN inputs according to the configs
        """
        self._current_milestep += 1

        # ------------------
        # BOUNDARY MASK:
        # ------------------
        # # For the moment, just pass the boundary mask:
        # if self._current_milestep!=0:
        #     current_segm = self.get_cache_data('cHC_data/img/node_labels', 'prev')
        # else:
        #     current_segm = self._init_segm
        #
        # # Use the cheap version for the moment:
        # # Only some affinities
        # bound_mask = rag_utils.compute_mask_boundaries(
        #     current_segm,
        #     self._offsets[1:3],
        #     channel_affs=0
        # )

        edge_indicators = self.get_cache_data('cHC_data/img/edge_indicators', 'prev') - 0.5


        # TEMP: pass pre-trained affinities:
        pretrained_affinities = self.get_cache_data('prediction_merge_prio', -1) - 0.5



        # TODO: pass ultra-contour-map previous steps
        # TODO: pass previous edge_indicators
        # TODO: pass normalized node_sizes


        # ------------------
        # Compute lookahead:
        # ------------------
        # TODO: implement look-aheads
        '''
         - rag on current_segm
         - get sizes nodes/edges
         - solve HC until termination (constrained if during training)
         - get nb of steps performed
         - (pass future-ultra-contour-image as a channel)
        '''
        # self._iterations_in_next_milestep = self.compute_lookahead()
        # self._iterations_in_next_milestep = -1

        # ------------------
        # Compose dictionary:
        # ------------------
        channels = {}
        # channels['boundary_mask'] = bound_mask
        channels['edge_indicators'] = edge_indicators
        channels['pretrained_affs'] = pretrained_affinities
        channels['raw'] = np.expand_dims(self._raw_image, axis=0)


        key_list = self._input_channels['dynamic_channels']
        return channels, key_list


    def collect_targets_and_weights_milestep(self):
        if self.validation_mode:
            empty_array = np.zeros_like(self.get_cache_data('cHC_data/img/edge_indicators', 'current'))
            return empty_array, empty_array
        else:
            merge_targets = self.get_cache_data('cHC_data/img/merge_targets', 'current')
            split_targets = self.get_cache_data('cHC_data/img/split_targets', 'current')
            loss_weights = self.get_cache_data('cHC_data/img/loss_weights', 'current')
            edge_indicators = self.get_cache_data('cHC_data/img/edge_indicators', 'prev')
            inner_label_ind = (edge_indicators < 0.)
            inner_label_targets = inner_label_ind.astype(np.float32)
            self.add_cache_data('cHC_data/img/inner_targets', inner_label_targets,'current')

            loss_weights[inner_label_ind] = self._weight_inner_label

            # merge_targets = loss_targ
            # split_targets = 1. - loss_targ
            # split_targets[np.where(loss_weigh==0)] = 0.

            total_targets = np.stack((merge_targets, split_targets, inner_label_targets), axis=1)
            loss_weights = np.expand_dims(loss_weights,axis=1)
            total_weights = np.concatenate((loss_weights,loss_weights,loss_weights), axis=1)
            # total_weights = np.ones_like(total_targets)

            # return loss_targ, loss_weigh
            return total_targets.astype(np.float32), total_weights.astype(np.float32)

    def build_clustering_from_init_segm(self, constrained,
                                        compute_loss_data,
                                        max_nb_milesteps=None,
                                        rag=None,
                                        init_segm=None,
                                        node_sizes=None,
                                        edge_sizes=None,
                                        GT_labels_nodes=None,
                                        affinities=None):
        """
        :param edge_sizes: The option "compute" is also accepted (automatically deduced from rag).
                In this case affinities option can be passed.
        """
        max_nb_milesteps = self._max_nb_milesteps if max_nb_milesteps is None else max_nb_milesteps
        if rag is None:
            if init_segm is None:
                rag = self._rag
                init_segm = self._init_segm
            else:
                rag = nrag.gridRag(init_segm.astype(np.uint32))
        else:
            assert init_segm is not None, "If rag is given, even init_segm is required"

        node_sizes = self.get_cache_data('cHC_data/vect/node_sizes', -1) if node_sizes is None else node_sizes
        if edge_sizes is None:
            edge_sizes = self.get_cache_data('cHC_data/vect/edge_sizes', -1)


        GT_labels_nodes = self.get_cache_data('cHC_data/vect/node_GT', -1) if GT_labels_nodes is None else GT_labels_nodes

        graph, is_local_edge = skunkworks.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering.utils.build_lifted_graph_from_rag(rag,
                                                                                                                                                    init_segm,
                                                                                                                                                    self._offsets,
                                                                                                                                                    max_lifted_distance=self._init_segm_opts.get('max_distance_lifted_edges', 3),
                                                                                                                                                    number_of_threads=self._nb_threads)

        # TODO: improve this shitty implementation please...
        return_edge_indicators = False
        return_edge_sizes = False
        edge_indicators = None
        if isinstance(edge_sizes, str):
            if edge_sizes == 'compute':
                return_edge_sizes = True
                if affinities is None:
                    affinities = np.zeros(init_segm.shape + (self._nb_offsets,), dtype=np.float)
                else:
                    return_edge_indicators = True
                edge_indicators, edge_sizes = \
                    skunkworks.postprocessing.segmentation_pipelines.features.accumulate_affinities_on_graph_edges(affinities, self._offsets,
                                                                                                                   graph=graph,
                                                                                                                   label_image=init_segm,
                                                                                                                   use_undirected_graph=True,
                                                                                                                   number_of_threads=self._nb_threads)
            else:
                raise NotImplementedError()

        clustering = skunkworks.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering.constrained_fixation_policy.constrained_fixation_clustering(
            graph,
            edge_sizes,
            node_sizes,
            init_segm.shape,
            is_local_edge,
            GT_labels_nodes=GT_labels_nodes,
            ignore_label=self._ignore_label,
            constrained=constrained,
            compute_loss_data=compute_loss_data,
            max_nb_milesteps=max_nb_milesteps,
            **self._fixation_kwargs
        )
        if not return_edge_sizes:
            return clustering, graph
        else:
            if return_edge_indicators:
                return clustering, graph, edge_indicators, edge_sizes
            else:
                return clustering, graph, edge_sizes

    def build_clustering_from_pixels(self, constrained,
                                        compute_loss_data,
                                        image_shape=None,
                                        max_nb_milesteps=None,
                                        GT_label_image=None):
        max_nb_milesteps = self._max_nb_milesteps if max_nb_milesteps is None else max_nb_milesteps
        GT_label_image = self._GT_label_image if GT_label_image is None else GT_label_image
        image_shape = self._raw_image.shape if image_shape is None else image_shape
        assert image_shape==GT_label_image.shape

        graph, is_local_edge, GT_labels_nodes = \
            skunkworks.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering.utils.build_pixel_lifted_graph_from_offsets(
                image_shape,
                self._offsets,
                GT_label_image=GT_label_image,
                nb_local_offsets=3
            )

        edge_sizes = np.ones(graph.numberOfEdges, dtype='float32')
        node_sizes = np.ones(graph.numberOfNodes, dtype='float32')

        clustering =  skunkworks.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering.constrained_fixation_policy.constrained_fixation_clustering(
            graph,
            edge_sizes,
            node_sizes,
            image_shape,
            is_local_edge,
            GT_labels_nodes=GT_labels_nodes,
            ignore_label=self._ignore_label,
            constrained=constrained,
            compute_loss_data=compute_loss_data,
            max_nb_milesteps=max_nb_milesteps,
            **self._fixation_kwargs
        )

        return clustering, graph, node_sizes, edge_sizes, GT_labels_nodes





    def compute_lookahead(self, edge_indicators):
        # Collect current data from cache:
        if self._current_milestep != 0:
            current_segm = self.get_cache_data('cHC_data/img/node_labels', 'prev')
            rag = nrag.gridRag(current_segm.astype(np.uint32))
        else:
            rag = self._rag
            current_segm = self._init_segm
        # Remark: in milestep>0 some nodes IDs are no longer in the segm,
        # but IDs in the new rag are consistent:
        node_sizes = self.get_cache_data('cHC_data/vect/node_sizes', 'prev')
        GT_labels_nodes = self.get_cache_data('cHC_data/vect/node_GT', 'prev')
        # edge_sizes = self.get_cache_data('cHC_data/vect/edge_sizes', 'prev')
        # edge_indicators = self.get_cache_data('cHC_data/vect/edge_indicators', 'prev')

        # Create lifted graph current segmentation:
        # TODO: change and redo agglomeration in nifty...?
        # Pro: no need to compute again edge_sizes, node_sizes, boundary_IDs, etc...
        # Cons: in some situation (start from pixels) nifty-clustering can be slow
        lookahead_clustering, _, _ = self.build_clustering_from_init_segm(
                                             constrained=not self.validation_mode,
                                             compute_loss_data=False,
                                             max_nb_milesteps=self._max_nb_milesteps - self._current_milestep,
                                             rag=rag,
                                             init_segm=current_segm,
                                             node_sizes=node_sizes,
                                             edge_sizes='compute',
                                             GT_labels_nodes=GT_labels_nodes)

        # Run clustering until threshold is reached:
        lookahead_clustering.run_next_milestep(edge_indicators,
                                               nb_iterations=-1)

        # Collect data about clustering:
        node_features, edge_features =  lookahead_clustering.get_all_last_data_milestep()

        # Deduce the total number of performed iterations:
        nb_iter_lookahead = lookahead_clustering._constrained_clustering.time() + 1
        nb_iterations_next_milestep = int(nb_iter_lookahead * 5. / 6.) + 1


        # Get rid of some features (not used atm):
        used_eKeys = deepcopy(eKeys)
        used_nKeys = deepcopy(nKeys)
        used_eKeys.pop('loss_targets')
        used_eKeys.pop('loss_weights')

        used_eKeys.pop('edge_sizes')
        used_eKeys.pop('edge_indicators')
        used_nKeys.pop('node_sizes')
        used_nKeys.pop('node_GT')

        # Map collected data back to image (used as additional CNN input):
        pass

        # print("Nb iterations in next milestep: ",nb_iterations_next_milestep)
        return nb_iterations_next_milestep

    def set_static_prediction(self, static_prediction):
        """
        If necessary, it generates the init_segmentation. Then rag is initialized.

        :param static_prediction: (channels,z,x,y) or (channels,x,y)
        """
        # ------------------
        # Generate init_segm:
        # ------------------
        if self._dims_prediction_model==2:
            static_prediction = np.expand_dims(static_prediction, axis=1)

        self.add_cache_data('prediction_merge_prio', static_prediction, 'current')

        # tick = time.time()
        init_segm = None
        if self._init_segm_opts.get('use_wsdt', False):
            init_segm, self._prob_map = self._wsdt(static_prediction)
        elif self._init_segm_opts.get('use_ws', False):
            init_segm, self._prob_map = self._ws(static_prediction)
        # print("WSDT/WS in {} s".format(time.time()-tick))

        if init_segm is not None:
            self.set_init_segm(init_segm)


        # ------------------
        # Init rag:
        # ------------------
        affinities = np.transpose(static_prediction, axes=(1, 2, 3, 0))
        self._init_long_range_graph(affinities)


    def _init_long_range_graph(self, initial_affinities):
        if self._training_data_available:
            assert self._GT_label_image is not None, "Set GT labels first!"
        # Constantine is using this label in the GT:
        self._ignore_label = 0
        constrained = not self.validation_mode
        compute_loss_data = False if not self._training_data_available or self.validation_mode else True

        if self._init_segm_opts['start_from_pixels']:
            # -----------------
            # Start from pixels:
            # -----------------
            assert self._init_segm is None, "Starting from pixels, init_segm should not be set"
            image_shape = np.array(self._HC_window_size)

            self._constrained_HC, self._graph, node_sizes, edge_sizes, GT_labels_nodes = \
                self.build_clustering_from_pixels(constrained,
                                              compute_loss_data,
                                              image_shape=image_shape,
                                              GT_label_image=self._GT_label_image)

            self._GT_labels_nodes = GT_labels_nodes

            if self._training_data_available:
                self._best_agglomeration = self._GT_label_image


            img_edge_indicators = np.transpose(initial_affinities, (3, 0, 1, 2))

            self._init_segm = self._graph.mapNodesIDToImage()
            self._init_boundMask_IDs = np.transpose(self._graph.mapEdgesIDToImage(), axes=(3, 0, 1, 2))
            raise NotImplementedError()
            # self.add_cache_data('cHC_data/vect/edge_indicators', edge_indicators, 'current')
        else:
            # -----------------
            # Start from superpixels:
            # -----------------
            assert self._init_segm is not None
            assert not self._init_segm_opts['relabel_continuous']

            # Find best GT targets:
            if self._training_data_available:
                # TODO: find a better bug-free solution:
                GT_border = ((self._GT_label_image == self._ignore_label).astype(np.int32) + 3) * 3
                self._init_segm = np.array(
                    vigra.analysis.labelMultiArray((self._init_segm * GT_border).astype(np.uint32)))

                self._GT_labels_nodes = find_best_agglomeration(self._init_segm, self._GT_label_image)
                self._best_agglomeration = (
                    segm_utils.map_features_to_label_array(
                        self._init_segm,
                        np.expand_dims(self._GT_labels_nodes, axis=-1),
                        number_of_threads=self._nb_threads)
                ).astype(np.int64)[...,0]

            self._rag = rag = nrag.gridRag(self._init_segm.astype(np.uint32))

            zero_affs = np.zeros(self._HC_window_size + (self._nb_offsets,), dtype=np.float)
            node_sizes = segm_utils.accumulate_segment_features_vigra(zero_affs[..., 0],
                                                                      self._init_segm, statistics=["Count"],
                                                                      normalization_mode=None, map_to_image=False)
            node_sizes = np.squeeze(node_sizes)

            self._constrained_HC, graph, edge_indicators, edge_sizes = \
                self.build_clustering_from_init_segm(
                    constrained=constrained,
                    compute_loss_data=compute_loss_data,
                    max_nb_milesteps=self._max_nb_milesteps,
                    rag=rag,
                    init_segm=self._init_segm,
                    node_sizes=node_sizes,
                    edge_sizes='compute',
                    GT_labels_nodes=self._GT_labels_nodes,
                    affinities=initial_affinities)
            self._graph = graph

            # Compute expensive boundary mask with edge IDs:
            bound_IDs = rag_utils.compute_mask_boundaries_graph(
                self._offsets,
                graph=graph,
                label_image=self._init_segm,
                return_boundary_IDs=True,
                channel_axis=0,
                use_undirected_graph=True,
                number_of_threads=self._nb_threads
            )
            self._init_boundMask_IDs = bound_IDs

            # self.add_cache_data('cHC_data/vect/edge_indicators', edge_indicators, 'current')
            img_edge_indicators = segm_utils.map_features_to_label_array(
                self._init_boundMask_IDs,
                np.expand_dims(edge_indicators, axis=-1),
                ignore_label=-1,
                fill_value=-1.,
                number_of_threads=self._nb_threads
            )[..., 0]


        # Save some data to cache:
        if self._training_data_available:
            self.add_cache_data('cHC_data/img/node_GT', self._GT_label_image, 'current')
            self.add_cache_data('cHC_data/vect/node_GT', self._GT_labels_nodes, 'current')

        self.add_cache_data('cHC_data/vect/node_sizes', node_sizes, 'current')
        self.add_cache_data('cHC_data/vect/edge_sizes', edge_sizes, 'current')
        self.add_cache_data('cHC_data/img/edge_indicators', img_edge_indicators,
                            'current')
        self.add_cache_data('cHC_data/vect/edge_indicators', edge_indicators, 'current')

    def run_clustering_on_pretrained_affs(self, start_from_pixels=False):
        scores = []
        for w in self.LHC:
            if not start_from_pixels:
                GT = w._GT_label_image
                clustering, _ = w.build_clustering_from_init_segm(False, False, max_nb_milesteps=-1)
                pretrained_edge_affs = w.get_cache_data('cHC_data/vect/edge_indicators', -1)
            else:
                GT = w._GT_label_image
                clustering, graph, _, _, _ = w.build_clustering_from_pixels(False, False, max_nb_milesteps=-1)
                pretrained_affs = w.get_cache_data('prediction', -1)
                pretrained_edge_affs = w.accumulate_affinities(pretrained_affs, pixel_grid_edges=True,
                                                               graph=graph)

            clustering.run_next_milestep(pretrained_edge_affs, nb_iterations=-1)
            # TODO: per carità, this should be moved into the clustering...
            final_segm = segm_utils.map_features_to_label_array(
                w._init_segm,
                np.expand_dims(clustering.current_data['node_labels'], axis=-1),
            number_of_threads=self._nb_threads
            )[...,0]

            scores.append(list(cremi_score(GT, final_segm,return_all_scores=True)))

        return np.mean(scores, axis=0)


    def set_raw_image(self, input):
        """ For the moment just raw image """
        assert input.shape==self._HC_window_size
        self._raw_image = input


    def set_init_segm(self, label_image, init_rag=False):
        assert self._init_segm is None, "Trying to set again the initial segmentation!"
        assert label_image.shape == self._HC_window_size
        self._init_segm = label_image # A possible remapping will happen in init_rag()
        if init_rag:
            self._init_rag()

    def set_targets(self, GT_labels):
        assert self._GT_label_image is None, "Trying to set again the GT segmentation!"
        assert GT_labels.shape == self._HC_window_size
        self._GT_label_image = GT_labels


    @ property
    def validation_mode(self):
        assert self._validation_mode is not None, "Validation mode was not set!"
        return self._validation_mode

    @validation_mode.setter
    def validation_mode(self, value):
        assert isinstance(value, bool)
        assert self._validation_mode is None, "Validation mode already set!"
        self._validation_mode = value

    def set_validation_mode(self, val_mode):
        self.validation_mode = val_mode
        self._HC_window_size = self._HC_window_size_valid if val_mode else self._HC_window_size_train

    def accumulate_affinities(self, prediction, pixel_grid_edges=None, graph=None):
        if pixel_grid_edges is None:
            pixel_grid_edges = self._init_segm_opts['start_from_pixels']
        else:
            assert isinstance(pixel_grid_edges,bool)

        graph = self._graph if graph is None else graph



        ndim = prediction.ndim
        assert ndim==3 or ndim==4

        if ndim==3 and self._dims_prediction_model==2:
            prediction = np.expand_dims(prediction, axis=1)

        affinities = np.transpose(prediction, axes=(1, 2, 3, 0))

        if pixel_grid_edges:
            edge_indicators = graph.edgeValues(affinities)
        else:
            edge_indicators, _ = \
                skunkworks.postprocessing.segmentation_pipelines.features.accumulate_affinities_on_graph_edges(affinities, self._offsets,
                                                                                                               graph=graph,
                                                                                                               label_image=self._init_segm,
                                                                                                               use_undirected_graph=True,
                                                                                                               number_of_threads=self._nb_threads)
        return edge_indicators

    def get_plot_images(self):
        assert self.is_finished()
        images = {}
        # Plot 1:
        images['raw'] = self._raw_image
        images['GT_labels'] = self._GT_label_image
        images['init_segm'] = self._init_segm
        images['prob_map'] = self._prob_map
        images['best_aggl'] = self._best_agglomeration
        images['pred_segm'] = self.get_cache_data('cHC_data/img/node_labels', 'current')
        images['final_UCM'] = self.get_cache_data('cHC_data/img/dendrogram_heigh', 'current')

        # Plot 2:
        images['predictions_merge_prio'] = np.stack(self.get_cache_data('prediction_merge_prio', list(range(-1,self._current_milestep+1))))
        images['predictions_notMerge_prio'] = np.stack(self.get_cache_data('prediction_notMerge_prio', 'all'))
        images['predictions_inner'] = np.stack(self.get_cache_data('prediction_inner', 'all'))
        images['edge_indicators'] = np.stack(self.get_cache_data('cHC_data/img/edge_indicators', list(range(-1,self._current_milestep+1))))
        if self._current_milestep == 0:
            images['predictions_notMerge_prio'] = np.expand_dims(images['predictions_notMerge_prio'], axis=0)
            images['predictions_inner'] = np.expand_dims(images['predictions_inner'], axis=0)
        # if self._current_milestep==0:
        #     images['edge_indicators'] = np.expand_dims(images['edge_indicators'], axis=0)
        if self._training_data_available and not self.validation_mode:
            # Only temporary for DEBUG:
            images['merge_targets'] = np.stack(self.get_cache_data('cHC_data/img/merge_targets', 'all'))
            images['split_targets'] = np.stack(self.get_cache_data('cHC_data/img/split_targets', 'all'))
            images['inner_targets'] = np.stack(self.get_cache_data('cHC_data/img/inner_targets', 'all'))
            images['loss_weights'] = np.stack(self.get_cache_data('cHC_data/img/loss_weights', 'all'))
            if self._current_milestep == 0:
                images['merge_targets'] = np.expand_dims(images['merge_targets'], axis=0)
                images['split_targets'] = np.expand_dims(images['split_targets'], axis=0)
                images['loss_weights'] = np.expand_dims(images['loss_weights'], axis=0)
                images['inner_targets'] = np.expand_dims(images['inner_targets'], axis=0)


        # Extra data:
        images["offsets"] = self._offsets
        images['prob_map_kwargs'] = self._init_segm_opts.get('prob_map_kwargs', {})

        if self.validation_mode:
            images['final_segm_pretrained'] = self.get_cache_data('final_segm_pretrained', 'current')
            images['scores'] = self.get_cache_data('scores', 'current')
            images['scores_pretrained'] = self.get_cache_data('scores_pretrained', 'current')

        return images


    def is_finished(self):
        return self._is_finished


    def clear(self):
        self._validation_mode = None
        self._cache_dict = {}
        self._raw_image = None
        self._GT_label_image = None
        self._init_segm = None
        self._init_boundMask_IDs = None
        self._bounding_boxes = None
        self._HC_window_size = None

        self._rag = None
        self._graph = None
        self._constrained_HC = None
        self._current_milestep = -1 # Static prediction is part of milestep==-1
        self._iterations_in_next_milestep = None
        self._is_finished = False

        self._GT_labels_nodes = None
        self._best_agglomeration = None
        self._ignore_label = None
        self._prob_map = None


    def get_cache_data(self,
                       name_data,
                       milestep,
                       from_disk=False,
                       as_array=True):
        """

        :param name_data:   string or list of strings.
        :param milestep:    None, integer, 'current', 'prev', list of integers or 'all' (only for one type of data)
        :param from_disk:   not supported atm
        :param as_array:    option for hdf5 data saved on disk

        :return Every requested array is a separated output
        """
        name_data = [name_data] \
            if isinstance(name_data, str) else name_data
        assert isinstance(name_data, list)

        if from_disk:
            raise NotImplementedError()

        if milestep is not None:
            if milestep=='current':
                milestep = self._current_milestep
            elif milestep=='prev':
                milestep = self._current_milestep - 1
            elif milestep=='all':
                assert len(name_data) == 1, "Multiple timesteps are supported only for one type of data"
                milestep = list(range(self._current_milestep+1))
            elif isinstance(milestep, list):
                assert len(name_data)==1, "Multiple timesteps are supported only for one type of data"
            else:
                assert isinstance(milestep, int)

        out = []
        for data in name_data:
            if not isinstance(milestep, list):
                internal_path = data if milestep is None else "{}/{}".format(data,milestep)
                assert internal_path in self._cache_dict, "Cache data not found! {}".format(internal_path)
                out.append(self._cache_dict[internal_path])
            else:
                for t in milestep:
                    internal_path = "{}/{}".format(data,t)
                    assert internal_path in self._cache_dict, "Cache data not found! {}".format(internal_path)
                    out.append(self._cache_dict[internal_path])
        if len(out)==1:
            return out[0]
        else:
            return tuple(out)

    def add_cache_data(self,
                       name_data,
                       data,
                       milestep,
                       from_disk=False,
                       as_array=True):
        """

        :param name_data:   string or list of strings
        :param data:        array or list of arrays
        :param milestep:    integer, 'current', 'prev'
        :param from_disk:   not supported atm
        :param as_array:    option for hdf5 data saved on disk
        :return:
        """
        name_data = [name_data] \
            if isinstance(name_data, str) else name_data
        assert isinstance(name_data, list)

        data = [data] if not isinstance(data, list) else data
        assert len(data)==len(name_data)

        if from_disk:
            raise NotImplementedError()

        if milestep is not None:
            if milestep=='current':
                milestep = self._current_milestep
            elif milestep=='prev':
                milestep = self._current_milestep - 1
            else:
                assert isinstance(milestep, int)

        for name_i, data_i in zip(name_data, data):
            internal_path = name_i if milestep is None else "{}/{}".format(name_i,milestep)
            # TODO: np.copy() can be expensive...
            self._cache_dict[internal_path] = np.copy(data_i)

    def __getstate__(self):
        state_dict = dict(self.__dict__)
        state_dict.pop('_rag')
        state_dict.pop('_graph')
        state_dict.pop('_constrained_HC')

        return state_dict

    def __setstate__(self, state_dict):
        # state_dict.setdefault('_rag', None)
        # raise NotImplementedError("Constrained HC needs to be manually restored")
        state_dict['_rag'] = None
        state_dict['_graph'] = None
        state_dict['_constrained_HC'] = None
        self.__dict__.update(state_dict)


class LHC(nn.Module):
    def __init__(self, options):
        super(LHC, self).__init__()
        self._batch_size = bs = options.get('HC_config').get('batch_size', 1)
        self.workers = [LHC_Worker(options) for _ in range(bs)]
        self.device_ids = [0] * bs
        self.output_device = self.device_ids[0]
        self.validation_mode = True
        self.unstructured_loss = None

    def apply_to_workers(self, function_name, *args, **kwargs):
        for w in self.workers:
            getattr(w, function_name)(*args, **kwargs)

    def pass_batch_data_to_workers(self, function_name, inputs):
        for i, input in enumerate(inputs):
            getattr(self.workers[i],function_name)(input)


    def forward(self, weights, **kwargs):
        # inputs = nn.parallel.scatter(weights.cpu(), self.device_ids)
        # cpu_w = weights.cpu()
        used_workers = self.workers[:weights.size()[0]]
        # TODO user parallel apply instead
        # outputs = nn.parallel.parallel_apply(used_workers, cpu_w)
        # return nn.parallel.gather(outputs, self.output_device).sum()
        results = [w(weights[i]) for i, w in enumerate(used_workers)]
        loss_targets = stack([results[i][0] for i in range(len(used_workers))])
        loss_weights = stack([results[i][1] for i in range(len(used_workers))])
        return loss_targets, loss_weights

    def run_clustering_on_pretrained_affs(self, start_from_pixels=False):
        """
        Used only during validation at the moment
        """
        if self.validation_mode:
            scores_pretrained = []
            scores = []
            for w in self.workers:
                # Compute scores dynamic affinities:
                score_worker_final = cremi_score(w._GT_label_image, w.get_cache_data('cHC_data/img/node_labels', 'current'),
                                           return_all_scores=True)
                w.add_cache_data('scores', score_worker_final, 'current')
                scores.append(list(score_worker_final))



                # Compute clustering on pre-trained affs:
                if not start_from_pixels:
                    GT = w._GT_label_image
                    clustering, _ = w.build_clustering_from_init_segm(False, False, max_nb_milesteps=-1)
                    pretrained_edge_affs = w.get_cache_data('cHC_data/vect/edge_indicators', -1)
                else:
                    GT = w._GT_label_image
                    clustering, graph, _, _, _ = w.build_clustering_from_pixels(False, False, max_nb_milesteps=-1)
                    pretrained_affs = w.get_cache_data('prediction', -1)
                    pretrained_edge_affs = w.accumulate_affinities(pretrained_affs, pixel_grid_edges=True,
                                                                   graph=graph)

                clustering.run_next_milestep(pretrained_edge_affs, nb_iterations=-1)
                # TODO: per carità, this should be moved into the clustering...
                final_segm = segm_utils.map_features_to_label_array(
                    w._init_segm,
                    np.expand_dims(clustering.current_data['node_labels'], axis=-1),
            number_of_threads=w._nb_threads
                )[...,0]
                w.add_cache_data('final_segm_pretrained', final_segm, 'current')
                scores_worker = cremi_score(GT, final_segm, return_all_scores=True)
                w.add_cache_data('scores_pretrained', scores_worker, 'current')

                scores_pretrained.append(list(scores_worker))

            self.final_score = np.mean(scores, axis=0)

    def set_static_input(self, inputs):
        """ For the moment just raw image """
        for i, input in enumerate(inputs):
            self.workers[i].set_static_input(input)


    def set_targets(self, targets):
        for i, t in enumerate(targets):
            self.workers[i].set_target(targets[i])

    def get_dynamic_inputs_milestep(self):
        dict_list = []
        key_list = None
        for w in self.workers:
            new_dict, key_list = w.get_dynamic_inputs_milestep()
            dict_list.append(new_dict)
        return dict_list, key_list

    def clear(self):
        self.final_score = None
        for w in self.workers:
            w.clear()

    def set_validation_mode(self, val_mode):
        self.validation_mode = val_mode
        for w in self.workers:
            w.set_validation_mode(val_mode)

    def get_plot_images(self):
        images = []
        for w in self.workers:
            images.append(w.get_plot_images())
        return images

    def is_finished(self):
        return np.all([w.is_finished() for w in self.workers])

    def __getitem__(self, key):
        return self.workers[key]
