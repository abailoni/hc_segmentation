import vigra
import numpy as np
from inferno.io.transform import Transform
from long_range_hc.criteria.learned_HC.utils.segm_utils import map_features_to_label_array
from long_range_hc.criteria.learned_HC.utils.rag_utils import compute_mask_boundaries

from long_range_hc.criteria.learned_HC.utils.segm_utils_CY import find_best_agglomeration
from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets

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
                 undersegm_threshold=None,
                 **super_kwargs):
        """
        :param ignore_label:
        :param border_thickness: Erode the GT labels and insert some ignore label on the boundary between segments.
        :param break_oversegm_on_GT_borders:
                Break oversegm segments on transitions to GT ignore labels (avoid to have huge segments that
                are labelled with the ignore label in the best_agglomeration)
        """
        self.ignore_label = ignore_label
        self.border_thickness = border_thickness
        self.number_of_threads = number_of_threads
        self.break_oversegm_on_GT_borders = break_oversegm_on_GT_borders
        self.return_node_labels_array = return_node_labels_array
        self.undersegm_threshold = undersegm_threshold
        super(FindBestAgglFromOversegmAndGT, self).__init__(**super_kwargs)

        self.offsets = None
        if border_thickness != 0:
            self.offsets = np.array([[0, border_thickness, 0],
                                [0, 0, border_thickness]])
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
                                                  undersegm_threshold=self.undersegm_threshold,
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
