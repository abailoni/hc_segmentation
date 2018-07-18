from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric import HDF5VolumeLoader, VolumeLoader
from inferno.utils import python_utils as pyu

class AffinitiesHDF5VolumeLoader(HDF5VolumeLoader):
    def __init__(self, path,
                 path_in_h5_dataset='data',
                 data_slice=None, name=None, dtype='float32',
                 **slicing_config):

        # Init super
        super(AffinitiesHDF5VolumeLoader, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                                              data_slice=data_slice, name=name,
                                                              **slicing_config)
        # Record attributes
        assert isinstance(dtype, str)
        self.dtype = dtype

    def __getitem__(self, index):
        # Casting to int would allow index to be IndexSpec objects.
        index = int(index)
        slices = self.base_sequence[index]
        sliced_volume = self.volume[tuple(slices)]
        if self.transforms is None:
            transformed = sliced_volume
        else:
            transformed = self.transforms(sliced_volume)

        if self.return_index_spec:
            raise NotImplementedError()
            return transformed, IndexSpec(index=index, base_sequence_at_index=slices)
        else:
            return transformed

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        path = config.get('path')
        path_in_h5_dataset = config.get('path_in_h5_dataset', None)
        name = config.get('name', None)
        dtype = config.get('dtype', 'float32')
        slicing_config = config.get('slicing_config', None)
        return cls(path,
                   path_in_h5_dataset=path_in_h5_dataset,
                   name=name, dtype=dtype,
                   **slicing_config)

class AffinitiesVolumeLoader(VolumeLoader):
    def __init__(self, volume,
                 name=None,
                 **slicing_config):
        slicing_config_for_name = pyu.get_config_for_name(slicing_config, name)

        # Init super
        super(AffinitiesVolumeLoader, self).__init__(volume=volume,
                                                     name=name,
                                        **slicing_config_for_name)

    def __getitem__(self, index):
        # Casting to int would allow index to be IndexSpec objects.
        index = int(index)
        slices = self.base_sequence[index]
        sliced_volume = self.volume[tuple(slices)]
        if self.transforms is None:
            transformed = sliced_volume
        else:
            transformed = self.transforms(sliced_volume)

        if self.return_index_spec:
            raise NotImplementedError()
            return transformed, IndexSpec(index=index, base_sequence_at_index=slices)
        else:
            return transformed

    @classmethod
    def from_config(cls, volume, name, config):
        config = yaml2dict(config)
        slicing_config = config.get('slicing_config', None)
        return cls(volume,
                   name=name,
                   **slicing_config)