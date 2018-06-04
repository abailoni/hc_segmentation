from copy import deepcopy

from inferno.io.core import ZipReject, Concatenate, Zip
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict
from inferno.utils import python_utils as pyu

from torch.utils.data.dataloader import DataLoader

from neurofire.datasets.cremi.loaders import SegmentationVolume
from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets
from neurofire.transform.volume import RandomSlide
from skunkworks.datasets.cremi.loaders.raw import RawVolumeWithDefectAugmentation
from skunkworks.transforms.artifact_source import RejectNonZeroThreshold

from long_range_hc.datasets.segm_transform import FromSegmToEmbeddingSpace


def get_multiple_datasets(dataset_names,
                          dataset_types,
                          volume_config,
                          slicing_config,
                          name=None,
                          defect_augmentation_config=None):
    """

    :param dataset_names: the keys expected in volume_config (e.g. 'raw', 'gt',
                'input_segm', etc...
    :param dataset_types: Accepted types: 'RawVolumeWithDefectAugmentation',
                                          'SegmentationVolume'
    :param volume_config:
    :param slicing_config:
    :param name: the name of the dataset (e.g. 'A' or so on. It can be left empty)
    :param defect_augmentation_config:
    :return:
    """
    dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
    dataset_types = dataset_types if isinstance(dataset_types, list) else [dataset_types]
    assert len(dataset_names) == len(dataset_types)

    assert isinstance(volume_config, dict)
    assert isinstance(slicing_config, dict)

    datasets = []
    for dt_name, dt_type in zip(dataset_names, dataset_types):
        assert dt_name in volume_config
        volume_kwargs = dict(volume_config.get(dt_name))
        volume_kwargs.update(slicing_config)

        if dt_type == 'RawVolumeWithDefectAugmentation':
            assert defect_augmentation_config is not None
            augmentation_config = deepcopy(defect_augmentation_config)

            if 'artifact_source' in augmentation_config:
                for slicing_key, slicing_item in augmentation_config['artifact_source']['slicing_config'].items():
                    if isinstance(slicing_item, dict):
                        new_item = augmentation_config['artifact_source']['slicing_config'][slicing_key][name]
                        augmentation_config['artifact_source']['slicing_config'][slicing_key] = new_item
            volume_kwargs.update({'defect_augmentation_config': augmentation_config})

            # Build raw volume:
            datasets.append(RawVolumeWithDefectAugmentation(name=name, **volume_kwargs))
        elif dt_type == 'SegmentationVolume':
            datasets.append(SegmentationVolume(name=name, **volume_kwargs))
        else:
            raise NotImplementedError("Passed key: {}".format(dt_type))

    datasets = datasets if len(datasets) != 1 else datasets[0]
    return datasets



class CREMIDatasetRealigned(Zip):
    def __init__(self, name, volume_config, slicing_config, defect_augmentation_config,
                 affinity_offsets,
                 inference_mode=False,
                 master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)

        self.inference_mode = inference_mode

        # FIXME: delete this
        slicing_config = deepcopy(slicing_config)
        slicing_config_affs = slicing_config.pop('slicing_config_affs', None)

        self.master_config = {} if master_config is None else master_config
        master_config = self.master_config

        self.apply_SegmToAff_to = []
        self.apply_FromSegmToEmbeddingSpace_to = []

        self.raw_volume = get_multiple_datasets('raw',
                                                  'RawVolumeWithDefectAugmentation',
                                                  volume_config,
                                                  slicing_config,
                                                  name=name,
                                                  defect_augmentation_config=defect_augmentation_config)
        list_of_datasets = [self.raw_volume]

        self.init_segm_volume = None
        self.init_boundaries_volume = None
        if 'init_segmentation' in volume_config:
            self.init_segm_volume = \
                get_multiple_datasets('init_segmentation',
                                      'SegmentationVolume',
                                      volume_config,
                                      slicing_config,
                                      name=name)
            list_of_datasets.append(self.init_segm_volume)
            # self.apply_SegmToAff_to.append(len(list_of_datasets)-1)
            self.apply_FromSegmToEmbeddingSpace_to.append(len(list_of_datasets) - 1)

        # if 'init_boundaries' in volume_config:
        #     assert slicing_config_affs is not None
        #     init_boundaries_kwargs = dict(volume_config.get('init_boundaries'))
        #     init_boundaries_kwargs.update(slicing_config_affs)
        #     # Build segmentation volume
        #     self.init_boundaries_volume = SegmentationVolume(name=name, **init_boundaries_kwargs)
        #     list_of_datasets.append(self.init_boundaries_volume)

        # Build segmentation volume

        self.GT_volume = None
        if not inference_mode:
            self.GT_volume = get_multiple_datasets('GT',
                                                   'SegmentationVolume',
                                                   volume_config,
                                                   slicing_config,
                                                   name=name)
            list_of_datasets.append(self.GT_volume)
            self.apply_SegmToAff_to.append(len(list_of_datasets) - 1)



        # Initialize zipreject:
        self.numb_of_datasets = len(list_of_datasets)
        super(CREMIDatasetRealigned, self).__init__(*list_of_datasets, sync=True)

        # FIXME: insert again the reject option for batches with only ignore label:
        # rejection_threshold = master_config.pop('rejection_threshold', 0.5)
        # super(CREMIDatasetRealigned, self).__init__(*list_of_datasets,
        #                                             sync=True, rejection_dataset_indices=len(list_of_datasets) - 1,
        #                                             rejection_criterion=RejectNonZeroThreshold(rejection_threshold))

        # Set master config (for transforms)
        self.affinity_offsets = affinity_offsets
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = None
        if not self.inference_mode:
            transforms = Compose(RandomFlip3D(),
                                 RandomRotate())

            # Elastic transforms can be skipped by setting elastic_transform to false in the
            # yaml config file.
            if self.master_config.get('elastic_transform'):
                elastic_transform_config = self.master_config.get('elastic_transform')
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

            # random slide augmentation
            if self.master_config.get('random_slides', False):
                assert self.init_segm_volume is None
                # TODO slide probability
                ouput_shape = self.master_config.get('shape_after_slide', None)
                max_misalign = self.master_config.get('max_misalign', None)
                transforms.add(RandomSlide(output_image_size=ouput_shape, max_misalign=max_misalign))

        # affs_on_gpu = self.master_config.get('affinities_on_gpu', False)
        # for_validation = self.master_config.get('for_validation', False)
        # if we compute the affinities on the gpu, or use the feeder for validation only,
        # we don't need to add the affinity transform here

        segm_to_aff = Segmentation2AffinitiesFromOffsets(dim=3,
                                                         offsets=pyu.from_iterable(self.affinity_offsets),
                                                         add_singleton_channel_dimension=True,
                                                         retain_segmentation=True,
                                                         apply_to=self.apply_SegmToAff_to)

        if transforms is None:
            transforms = Compose(segm_to_aff)
        else:
            transforms.add(segm_to_aff)


        # Next: crop invalid affinity labels and
        # elastic augment reflection padding assymetrically
        if not self.inference_mode:
            crop_config = self.master_config.get('crop_after_target', {})
            if crop_config:
                # One might need to crop after elastic transform
                # to avoid edge artefacts of affinity
                # computation being warped into the FOV.
                transforms.add(VolumeAsymmetricCrop(**crop_config))

        transforms.add(FromSegmToEmbeddingSpace(dim_embedding_space=12,
                                                number_of_threads=8,
                            apply_to=self.apply_FromSegmToEmbeddingSpace_to))

        return transforms

    @classmethod
    def from_config(cls, config, inference_mode=False):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        offsets = config['offsets']
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   inference_mode=inference_mode,
                   affinity_offsets=offsets,
                   defect_augmentation_config=defect_augmentation_config,
                   master_config=master_config)


class CREMIDatasetsRealigned(Concatenate):
    def __init__(self, names, volume_config, slicing_config,
                 defect_augmentation_config,
                 master_config=None):
        # Make datasets and concatenate
        datasets = [CREMIDatasetRealigned(name=name,
                                          volume_config=volume_config,
                                          slicing_config=slicing_config,
                                          defect_augmentation_config=defect_augmentation_config,
                                          master_config=master_config)
                    for name in names]
        # Concatenate
        super(CREMIDatasetsRealigned, self).__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('dataset_names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        return cls(names=names, volume_config=volume_config,
                   defect_augmentation_config=defect_augmentation_config,
                   slicing_config=slicing_config, master_config=master_config)



def get_cremi_loaders_realigned(config):
    """
    Gets CREMI Loaders given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    datasets = CREMIDatasetsRealigned.from_config(config)
    loader = DataLoader(datasets, **config.get('loader_config'))
    return loader
