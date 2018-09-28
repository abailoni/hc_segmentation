import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os

import numpy as np

from long_range_hc.postprocessing.data_utils import import_dataset, import_segmentations, import_SOA_datasets

from cremi import Annotations, Volume
from cremi.io import CremiFile




pad_first_block = {
    'A': ((0, 0), (22, 0), (0, 94)),
    'B': ((0, 0), (48, 0), (0, 224)),
    # 'C': ((0, 0), (12, 0), (0, 12)),
}
mis_slice = {
    'A': 117,  # In the padded coordinates
    'B': 53,
    # 'C': 113
}
# They will be "blacked" (in the padded frame):
defected_slices = {
    'A': [25, 37, 70],
    'C': [123]
}

wanted_pad = ((10, 10), (200, 200), (200, 200))
original_pad = ((37, 38), (911, 911), (911, 911))
padded_shape = (200, 3072, 3072)

def apply_initial_pad(volume, offset):
    offset = (0, 0, 0) if offset is None else offset
    initial_pad = []
    for i in range(3):
        diff = padded_shape[i] - offset[i] - volume.shape[i]
        assert diff >= 0
        initial_pad.append((offset[i], diff))
    return np.pad(volume, tuple(initial_pad), mode='constant')


# Usual offset: (27, 711, 711)

def align(volume, sample, offset=None,
          add_black_slices=True,
          align_slices=True):
    volume = apply_initial_pad(volume, offset)


    # Make some of the defected slices black:
    vol_transf = volume
    if sample in defected_slices and add_black_slices:
        for z_slc in defected_slices[sample]:
            vol_transf[z_slc] = 0

    # Align (if necessary):
    if sample in pad_first_block and align_slices:
        pad = pad_first_block[sample]
        inverse_pad = tuple([(pd[1], pd[0]) for pd in pad])
        raw1 = np.pad(vol_transf[:mis_slice[sample]], pad, mode='constant')
        raw2 = np.pad(vol_transf[mis_slice[sample]:], inverse_pad, mode='constant')
        vol_transf = np.concatenate([raw1, raw2], axis=0)
        # Crop extra added pad symmetrically:
        assert all([(pd[0] + pd[1]) % 2 == 0 for pd in pad]), "Pads should be even"
        crops = [(pd[0] + pd[1]) / 2 for pd in pad]
        crops_slice = []
        for cr in crops:
            if cr != 0:
                crops_slice.append(slice(cr, -cr))
            else:
                crops_slice.append(slice(None))
        vol_transf = vol_transf[tuple(crops_slice)]

    # Crop extra padding:
    pad_crop = tuple([slice(pd2[0] - pd1[0], -(pd2[1] - pd1[1])) for pd1, pd2 in zip(wanted_pad, original_pad)])
    raw_transf_cropped = vol_transf[pad_crop]
    return raw_transf_cropped


def undo_alignment(volume, sample, offset=None):
    # volume = apply_initial_pad(volume, offset)

    # # Unalign for submission:
    vol_unaligned = volume
    if sample in pad_first_block:
        pad = pad_first_block[sample]
        z_slice = mis_slice[sample] - (original_pad[0][0] - wanted_pad[0][0])
        crop_slc_1 = (slice(0,z_slice), slice(pad[1][0], -pad[1][1] if pad[1][1]!=0 else None), slice(pad[2][0], -pad[2][1] if pad[2][1]!=0 else None))
        crop_slc_2 = (slice(z_slice, None), slice(pad[1][1], -pad[1][0] if pad[1][0]!=0 else None), slice(pad[2][1], -pad[2][0] if pad[2][0]!=0 else None))
        vol_unaligned = np.concatenate((vol_unaligned[crop_slc_1], vol_unaligned[crop_slc_2]))
    final_shape = (125, 1250, 1250)
    wanted_pad_slc = [slice(int((shp1-shp2)/2), int(-(shp1-shp2)/2)) for shp1, shp2 in zip(vol_unaligned.shape, final_shape)]
    print(wanted_pad_slc)
    return vol_unaligned[tuple(wanted_pad_slc)]



project_folder = '/export/home/abailoni/learnedHC/model_050_A_v3/pureDICE_wholeDtSet'
# aggl_name_partial = 'inferName_v100k-alignedTestOversegmPlusMC_HC065_'
aggl_name_partial = 'inferName_v100k-alignedTestOversegmPlusMC_MCfull080_'
key = 'finalSegm_WS'


for sample in [
    # 'B',
    # 'C',
    'A',
]:
    aggl_name = aggl_name_partial + sample

    segm = import_segmentations(project_folder, aggl_name,
                                             keys_to_return=[key])

    cropped_segm = undo_alignment(segm, sample)
    # crop_slc = (slice(10, -10), slice(200, -200), slice(200, -200))
    # cropped_segm = segm[crop_slc]
    cropped_segm = vigra.analysis.labelVolume(cropped_segm.astype('uint32'))


    file_path = os.path.join(project_folder, "postprocess/{}/pred_segm.h5".format(aggl_name))
    vigra.writeHDF5(cropped_segm, file_path, 'finalSegm_submission', compression='gzip')

    print("Writing results...")
    # Open a file for writing (deletes previous file, if exists)
    cremi_path = os.path.join(project_folder, "postprocess/{}/submission.hdf".format(aggl_name))
    file = CremiFile(cremi_path, "w")

    # Write volumes representing the neuron and synaptic cleft segmentation.
    neuron_ids = Volume(cropped_segm.astype('uint64'), resolution=(40.0, 4.0, 4.0), comment="SP-CNN-submission")

    file.write_neuron_ids(neuron_ids)

    file.close()
