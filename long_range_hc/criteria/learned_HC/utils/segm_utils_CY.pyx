# import numpy as np

cimport numpy as np
import numpy as np

# TODO: find better way to combine the different versions...

cdef np.ndarray[long, ndim=4] map_features_to_label_array_CY_3D(np.ndarray[long, ndim=3] label_image,
                                                             np.ndarray[double, ndim=2] feature_array,
                                                             long ignore_label,
                                                             double fill_value):
    cdef:
        int dim0 = label_image.shape[0]
        int dim1 = label_image.shape[1]
        int dim2 = label_image.shape[2]
        int nb_feat = feature_array.shape[1]
    feature_image = np.ones((dim0, dim1, dim2, feature_array.shape[1])) * fill_value

    cdef double[:,:,:,::1] feature_image_c = feature_image
    cdef double[:,::1] feature_array_c = feature_array
    cdef long[:,:,::1] label_image_c = label_image


    cdef long label

    for k in range(dim0):
        for i in range(dim1):
            for j in range(dim2):
                label = label_image_c[k,i,j]
                if label!=ignore_label:
                    for f in range(nb_feat):
                        feature_image_c[k,i,j,f] = feature_array_c[label, f]

    return feature_image

cdef np.ndarray[long, ndim=5] map_features_to_label_array_CY_4D(np.ndarray[long, ndim=4] label_image,
                                                             np.ndarray[double, ndim=2] feature_array,
                                                             long ignore_label,
                                                             double fill_value):
    cdef:
        int dim0 = label_image.shape[0]
        int dim1 = label_image.shape[1]
        int dim2 = label_image.shape[2]
        int dim3 = label_image.shape[3]
        int nb_feat = feature_array.shape[1]

    feature_image = np.ones((dim0, dim1, dim2, dim3, feature_array.shape[1])) * fill_value


    cdef double[:,:,:,:,::1] feature_image_c = feature_image
    cdef double[:,::1] feature_array_c = feature_array
    cdef long[:,:,:,::1] label_image_c = label_image

    cdef long label

    for k in range(dim0):
        for i in range(dim1):
            for j in range(dim2):
                for t in range(dim3):
                    label = label_image_c[k,i,j,t]
                    if label!=ignore_label:
                        for f in range(nb_feat):
                            feature_image_c[k,i,j,t,f] = feature_array_c[label, f]

    return feature_image

def map_features_to_label_array(label_image, feature_array, ignore_label=None, fill_value=0.):
    """
    feature_array:

        - first dimension gives ID of the labels
        - second dimension represents the different features

    """
    # TODO: this could be expensive:
    label_image = label_image.copy(order='C')
    feature_array = feature_array.copy(order='C')

    ignore_label = -1 if ignore_label is None else ignore_label
    if label_image.ndim==3:
        return map_features_to_label_array_CY_3D(label_image.astype(np.int64), feature_array.astype(np.float64), ignore_label, fill_value)
    elif label_image.ndim==4:
        return map_features_to_label_array_CY_4D(label_image.astype(np.int64), feature_array.astype(np.float64), ignore_label, fill_value)
    else:
        raise NotImplementedError()



cdef np.ndarray[long, ndim=1] find_best_agglomeration_CY(np.ndarray[long, ndim=3] segm, np.ndarray[long, ndim=3] GT_segm,
                                                         long undersegm_threshold,
                                                         long ignore_label):
    shape = segm.shape
    max_segm, max_GT  = (segm.max()+1).astype(np.uint64), (GT_segm.max()+1).astype(np.uint64)
    inter_matrix = np.zeros((max_segm, max_GT), dtype=np.uint32)
    flat_segm, flat_GT = segm.flatten().astype(np.uint64), GT_segm.flatten().astype(np.uint64)

    cdef unsigned long[::1] flat_segm_c = flat_segm
    cdef unsigned long[::1] flat_GT_c = flat_GT
    cdef unsigned int[:,::1] inter_matrix_c = inter_matrix
    cdef int dim0 = flat_GT.shape[0]

    for i in range(dim0):
        inter_matrix_c[flat_segm_c[i], flat_GT_c[i]] += 1

    best_labels = np.argmax(inter_matrix, axis=1)
    if undersegm_threshold != 0:
        segm_mask = inter_matrix >= undersegm_threshold
        segm_mask[:, ignore_label] = False
        best_labels[np.sum(segm_mask, axis=1) > 1] = ignore_label
    return best_labels



def find_best_agglomeration(segm, GT_segm, undersegm_threshold=None, ignore_label=None):
    assert segm.ndim == 3, "Only 3D at the moment"
    assert segm.shape == GT_segm.shape
    assert segm.min() >= 0 and GT_segm.min() >= 0, "Only positive labels are expected"

    if undersegm_threshold is None:
        undersegm_threshold = 0
    if ignore_label is None:
        ignore_label = 0

    return find_best_agglomeration_CY(segm.astype(np.int64), GT_segm.astype(np.int64),
                                      undersegm_threshold, ignore_label)