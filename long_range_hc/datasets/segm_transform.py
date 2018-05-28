import vigra
import numpy as np
from inferno.io.transform import Transform
from long_range_hc.criteria.learned_HC.utils.segm_utils import map_features_to_label_array


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

    def tensor_function(self, tensor):
        """
        Expected shape: (1 , z, x, y)

        Label 0 represents ignore-label (often boundary between segments).
        """
        tensor = tensor.astype(np.uint32)

        # This keep the 0-label intact and starts from 1:
        tensor, max_label, _ = vigra.analysis.relabelConsecutive(tensor)

        self.build_random_variables(num_segments=max_label+1)
        embedding_vectors = self.get_random_variable('embedding_vectors')

        embedded_tensor = map_features_to_label_array(tensor,embedding_vectors,
                                    ignore_label=0,
                                    fill_value=0.,
                                    number_of_threads=self.number_of_threads)

        # Normalize values:
        embedded_tensor = (embedded_tensor - embedded_tensor.mean()) / embedded_tensor.std()
        embedded_tensor = np.rollaxis(embedded_tensor, axis=-1, start=0)

        if self.keep_segm:
                embedded_tensor = np.concatenate((np.expand_dims(tensor, axis=0),
                                                  embedded_tensor))

        # Expand dimension:
        return embedded_tensor