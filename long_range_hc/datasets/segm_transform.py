import vigra
import numpy as np
from inferno.io.transform import Transform
import inferno.utils.python_utils as pyu
from long_range_hc.criteria.learned_HC.utils.rag_utils import compute_mask_boundaries, map_edge_features_to_image

from long_range_hc.criteria.learned_HC.utils.segm_utils_CY import find_best_agglomeration
from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets
from long_range_hc.criteria.learned_HC.utils.segm_utils_CY import find_split_GT, cantor_pairing_fct
from long_range_hc.criteria.learned_HC.utils.segm_utils import accumulate_segment_features_vigra, map_features_to_label_array, cantor_pairing_fct

import nifty.graph.rag as nrag

from neurofire.transform.segmentation import get_boundary_offsets

class FindBestAgglFromOversegmAndGT(Transform):
    """
    Given an initial segm. and some GT labels, it finds the best agglomeration that can be done to
    get as close as possible to the GT labels.
    """
    def __init__(self, ignore_label=0,
                 border_thickness=0,
                 number_of_threads=8,
                 break_oversegm_on_GT_borders=False,
                 return_node_labels_array=False,
                 undersegm_rel_threshold=None,
                 **super_kwargs):
        """
        :param ignore_label:
        :param border_thickness: Erode the GT labels and insert some ignore label on the boundary between segments.
        :param break_oversegm_on_GT_borders:
                Break oversegm segments on transitions to GT ignore labels (avoid to have huge segments that
                are labelled with the ignore label in the best_agglomeration)
        :param undersegm_rel_threshold:
                The best matching GT label should cover at least this relative percentage of the segment, otherwise
                we consider it undersegmentated and we label it with the ignore label.
                E.g. 0.7 means: the best matching GT label should be at least 70% of the segment.
        """
        self.ignore_label = ignore_label
        self.border_thickness = border_thickness
        self.number_of_threads = number_of_threads
        self.break_oversegm_on_GT_borders = break_oversegm_on_GT_borders
        self.return_node_labels_array = return_node_labels_array
        self.undersegm_rel_threshold = undersegm_rel_threshold
        super(FindBestAgglFromOversegmAndGT, self).__init__(**super_kwargs)

        self.offsets = None
        if border_thickness != 0:
            self.offsets = np.array(get_boundary_offsets([0,border_thickness,border_thickness]))
            # self.get_border_mask = Segmentation2AffinitiesFromOffsets(3,
            #                                                       offsets=[[0,border_thickness,0],
            #                                                                [0,0,border_thickness]],
            #                                                       add_singleton_channel_dimension=True)

    def batch_function(self, tensors):
        init_segm, GT_labels = tensors

        if self.break_oversegm_on_GT_borders:
            # TODO: find a better bug-free solution:
            GT_border = ((GT_labels == self.ignore_label).astype(np.int32) + 3) * 3
            init_segm = np.array(
                vigra.analysis.labelMultiArray((init_segm * GT_border).astype(np.uint32)))
        else:
            init_segm, _, _ = vigra.analysis.relabelConsecutive(init_segm.astype('uint32'))

        if self.ignore_label == 0:
            # This keeps the zero label:
            GT_labels, _, _ = vigra.analysis.relabelConsecutive(GT_labels.astype('uint32'))
        else:
            raise NotImplementedError()


        if self.border_thickness != 0:

            border_affs = 1- compute_mask_boundaries(GT_labels,
                                                  self.offsets,
                                                  compress_channels=False,
                                                  channel_affs=0)
            border_mask = np.logical_and(border_affs[0], border_affs[1])
            if self.ignore_label == 0:
                GT_labels *= border_mask
            else:
                GT_labels[border_mask==0] = self.ignore_label

        GT_labels_nodes = find_best_agglomeration(init_segm, GT_labels,
                                                  undersegm_rel_threshold=self.undersegm_rel_threshold,
                                                  ignore_label=self.ignore_label)
        if self.return_node_labels_array:
            return GT_labels_nodes

        best_agglomeration = (
            map_features_to_label_array(
                init_segm,
                np.expand_dims(GT_labels_nodes, axis=-1),
                number_of_threads=self.number_of_threads)
        ).astype(np.int64)[...,0]

        return best_agglomeration


class FindSplitGT(Transform):
    def __init__(self,
                 size_small_segments_rel,
                 ignore_label=0,
                 border_thickness_GT=0,
                 border_thickness_segm=0,
                 number_of_threads=8,
                 break_oversegm_on_GT_borders=False,
                 **super_kwargs):
        self.ignore_label = ignore_label
        self.border_thickness_GT = border_thickness_GT
        self.border_thickness_segm = border_thickness_segm
        self.number_of_threads = number_of_threads
        self.break_oversegm_on_GT_borders = break_oversegm_on_GT_borders
        self.size_small_segments_rel = size_small_segments_rel
        super(FindSplitGT, self).__init__(**super_kwargs)

        self.offsets = None
        if border_thickness_GT != 0:
            self.offsets_GT = np.array(get_boundary_offsets([0,border_thickness_GT,border_thickness_GT]))
        if border_thickness_segm != 0:
            self.offsets_segm = np.array(get_boundary_offsets([0,border_thickness_segm,border_thickness_segm]))

    def batch_function(self, tensors):
        finalSegm, GT_labels = tensors

        ignore_mask = GT_labels == self.ignore_label
        if self.break_oversegm_on_GT_borders:
            # TODO: find a better bug-free solution:
            finalSegm = np.array(
                vigra.analysis.labelMultiArray((finalSegm * ((ignore_mask.astype(np.int32) + 3) * 3)).astype(np.uint32)))
        else:
            finalSegm, _, _ = vigra.analysis.relabelConsecutive(finalSegm.astype('uint32'))

        if self.ignore_label == 0:
            # This keeps the zero label:
            GT_labels, _, _ = vigra.analysis.relabelConsecutive(GT_labels.astype('uint32'))
        else:
            raise NotImplementedError()


        if self.border_thickness_GT != 0:

            border_affs = 1- compute_mask_boundaries(GT_labels,
                                                  self.offsets_GT,
                                                  compress_channels=False,
                                                  channel_affs=0)
            border_mask = np.logical_and(border_affs[0], border_affs[1])
            if self.ignore_label == 0:
                GT_labels *= border_mask
            else:
                GT_labels[border_mask==0] = self.ignore_label

        # Erode also oversegmentation:
        if self.border_thickness_segm != 0:
            border_affs = 1 - compute_mask_boundaries(finalSegm,
                                                      self.offsets_segm,
                                                      compress_channels=False,
                                                      channel_affs=0)
            border_mask = np.logical_and(border_affs[0], border_affs[1])
            if self.ignore_label == 0:
                GT_labels *= border_mask
            else:
                GT_labels[border_mask == 0] = self.ignore_label

        split_GT = find_split_GT(finalSegm, GT_labels,
                                                  size_small_segments_rel=self.size_small_segments_rel,
                                                  ignore_label=self.ignore_label)
        # split_GT = GT_labels


        if True:
            new_split_GT = np.zeros_like(split_GT)
            for z in range(split_GT.shape[0]):
                z_slice = split_GT[[z]].astype(np.uint32)
                z_slice_compontents = np.array(
                    vigra.analysis.labelMultiArrayWithBackground(z_slice, background_value=self.ignore_label))
                sizeMap = accumulate_segment_features_vigra([z_slice_compontents],
                                                            [z_slice_compontents],
                                                            ['Count'],
                                                            ignore_label=0,
                                                            map_to_image=True
                                                            ).squeeze(axis=-1)

                z_slice[sizeMap <= 50] = self.ignore_label

                # WS nonsense:
                mask_for_WS = compute_mask_boundaries(finalSegm[[z]],
                                        np.array(
                                            get_boundary_offsets([0, 1, 1])),
                                        compress_channels=True)
                mask_for_WS = - vigra.filters.boundaryDistanceTransform(mask_for_WS.astype('float32'))
                mask_for_WS = np.random.normal(scale=0.001, size=mask_for_WS.shape) + mask_for_WS
                mask_for_WS += abs(mask_for_WS.min())


                mask_for_eroding_GT = 1 - compute_mask_boundaries(finalSegm[[z]],
                                        np.array(
                                            get_boundary_offsets([0, 8, 8])),
                                        compress_channels=True)
                seeds = (z_slice + 1) * mask_for_eroding_GT

                z_slice, _ = vigra.analysis.watershedsNew(mask_for_WS[0].astype('float32'), seeds=seeds[0].astype('uint32'),
                                                                     method='RegionGrowing')

                new_split_GT[z] = z_slice - 1
            split_GT = new_split_GT

        split_GT = (split_GT + 1) * (1 - ignore_mask)

        return split_GT


class ComputeStructuredWeightsWrongMerges(Transform):
    def __init__(self,
                 offsets,
                 dim=3,
                 ignore_label=0,
                 number_of_threads=8,
                 weighting=1.0,
                 **super_kwargs):
        """

        :param offsets:
        :param dim:
        :param ignore_label:
        :param number_of_threads:
        :param weighting: max is 1.0, min is 0.0 (this function has no effect and all weights are 1.0)
        :param super_kwargs:
        """
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        assert len(offsets) > 0, "`offsets` must not be empty."
        assert ignore_label >= 0

        assert dim in (2, 3), "Affinities are only supported for 2d and 3d input"

        self.offsets = np.array(offsets)
        self.ignore_label = ignore_label
        self.weighting = weighting
        self.dim = dim
        self.number_of_threads = number_of_threads
        super(ComputeStructuredWeightsWrongMerges, self).__init__(**super_kwargs)


    def batch_function(self, tensors):
        # TODO: add check for the ignore label!!
        finalSegm, GT_labels = tensors

        intersection_segm = cantor_pairing_fct(finalSegm, GT_labels)
        intersection_segm, max_label, _ = vigra.analysis.relabelConsecutive(intersection_segm.astype('uint32'))

        rag = nrag.gridRag(intersection_segm, numberOfThreads=self.number_of_threads)

        _, node_features = nrag.accumulateMeanAndLength(rag=rag, data=GT_labels.astype('float32'),
                                        numberOfThreads=self.number_of_threads)

        size_nodes = node_features[:,1].astype('int')
        GT_labels_nodes = node_features[:,0].astype('int')

        _, node_features = nrag.accumulateMeanAndLength(rag=rag, data=finalSegm.astype('float32'),
                                                           numberOfThreads=self.number_of_threads)
        segm_labels_nodes = node_features[:,0].astype('int')


        uv_ids = rag.uvIds()

        wrong_merge_condition = np.logical_and(GT_labels_nodes[uv_ids[:,0]] != GT_labels_nodes[uv_ids[:,1]],
                                   segm_labels_nodes[uv_ids[:, 0]] == segm_labels_nodes[uv_ids[:, 1]])
        edge_weights = np.where(wrong_merge_condition,
                             1 + np.minimum(size_nodes[uv_ids[:,0]], size_nodes[uv_ids[:,1]]) * self.weighting,
                             np.ones(uv_ids.shape[0]))

        loss_weights = map_edge_features_to_image(self.offsets, np.expand_dims(edge_weights, -1), rag=rag,
                                   channel_affs=0, fillValue=1.,
                                   number_of_threads=self.number_of_threads)[...,0]

        return loss_weights




class FromSegmToEmbeddingSpace(Transform):
    def __init__(self, dim_embedding_space=12,
                 number_of_threads=8,
                 keep_segm=True,
                 **super_kwargs):
        self.dim_embedding_space = dim_embedding_space
        self.number_of_threads = number_of_threads
        self.keep_segm = keep_segm
        super(FromSegmToEmbeddingSpace, self).__init__(**super_kwargs)

    def build_random_variables(self, num_segments=None):
        np.random.seed()
        assert isinstance(num_segments, int)
        self.set_random_variable('embedding_vectors',
                np.random.uniform(size=(num_segments, self.dim_embedding_space)))

    def tensor_function(self, tensor_):
        """
        Expected shape: (z, x, y) or (channels , z, x, y)

        Label 0 represents ignore-label (often boundary between segments).

        If channels are passed, at the moment:
            - labels are expected as fist channel
            - it returns labels-EmbeddingVectors-previouslyPassedChannels
        """
        def convert_tensor(tensor, max_label = None):
            tensor = tensor.astype(np.uint32)

            if max_label is None:
                max_label = tensor.max()

            self.build_random_variables(num_segments=max_label+1)
            embedding_vectors = self.get_random_variable('embedding_vectors')

            embedded_tensor = map_features_to_label_array(tensor,embedding_vectors,
                                        ignore_label=0,
                                        fill_value=0.,
                                        number_of_threads=self.number_of_threads)

            # Normalize values:
            embedded_tensor = (embedded_tensor - embedded_tensor.mean()) / embedded_tensor.std()

            # TODO: improve!
            if tensor.ndim == 3:
                embedded_tensor = np.rollaxis(embedded_tensor, axis=-1, start=0)

                if self.keep_segm:
                        embedded_tensor = np.concatenate((np.expand_dims(tensor, axis=0),
                                                          embedded_tensor))
            elif tensor.ndim == 4:
                embedded_tensor = embedded_tensor[...,0]

            # Expand dimension:
            return embedded_tensor.astype('int32')

        if tensor_.ndim == 3:
            # This keep the 0-label intact and starts from 1:
            tensor_continuous, max_label, _ = vigra.analysis.relabelConsecutive(tensor_.astype('uint32'))
            out =  convert_tensor(tensor_continuous, max_label)
            return out
        elif tensor_.ndim == 4:
            labels = tensor_[0]
            labels_continuous, max_label, _ = vigra.analysis.relabelConsecutive(labels.astype('uint32'))
            vectors = convert_tensor(labels_continuous, max_label)
            out = np.concatenate((vectors, tensor_[1:]), axis=0)
            return out
        else:
            raise NotImplementedError()
