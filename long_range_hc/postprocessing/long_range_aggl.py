import numpy as np

from inferno.utils.io_utils import yaml2dict
from skunkworks.inference.simple import SimpleParallelLoader
import vigra

# TODO: upgrade this basic version with something more advanced
# here the bad thing is that in the final agglomeration all boundaries between blocks are hard vertical/horizontal boundaries

from skunkworks.postprocessing.segmentation_pipelines.base import SegmentationPipeline
from .segmentation_pipelines.agglomeration.fixation_clustering.pipelines import FixationAgglomeraterFromSuperpixels

class BlockWise(object):
    def __init__(self, segmentation_pipeline,
                 offsets,
                 blockwise=True,
                 final_agglomerater=None,
                 invert_affinities=False, # Only used for final aggl.
                 nb_threads=8,
                 return_fragments=False,
                 blockwise_config=None):
        self.segmentation_pipeline = segmentation_pipeline
        self.blockwise = blockwise
        self.return_fragments = return_fragments
        if blockwise:
            assert blockwise_config is not None
            self.blockwise_solver = BlockWiseSegmentationPipelineSolver.from_config(
                segmentation_pipeline,
                offsets,
                blockwise_config)

            # At the moment the final agglomeration is a usual  HC with 0.5 threshold:
            if final_agglomerater is None:
                raise DeprecationWarning()
                self.final_agglomerater = FixationAgglomeraterFromSuperpixels(
                    offsets,
                    max_distance_lifted_edges=2,
                    update_rule_merge='mean',
                    update_rule_not_merge='mean',
                    zero_init=False,
                    n_threads=nb_threads,
                    invert_affinities=invert_affinities)
            else:
                self.final_agglomerater = final_agglomerater

    def __call__(self, input_):
        # TODO: check that input_ is a dataset
        final_crop = tuple(slice(pad[0], input_.volume.shape[i+1] - pad[1]) for i, pad in enumerate(input_.padding[1:]))
        if self.blockwise:
            # TODO: change this!!
            # At the moment if we crop the padding, then we need to crop the global border for the final
            # agglomeration (but in this way we lose affinities context).
            # The alternative is to keep somehow the segmentation on the borders...
            blockwise_segm = self.blockwise_solver(input_)
            if self.return_fragments:
                fragments = blockwise_segm[0]
                blockwise_segm = blockwise_segm[1]

            # ---- TEMP ----
            blockwise_segm = blockwise_segm[final_crop]
            affs = input_.volume[(slice(None),) + final_crop]
            # ---- TEMP ----

            output_segm = self.final_agglomerater(affs, blockwise_segm)
            # output_segm = output_segm[final_crop]
            # blockwise_segm = blockwise_segm[final_crop]
            if self.return_fragments:
                return output_segm, blockwise_segm, fragments
            else:
                return output_segm, blockwise_segm
        else:
            output_segm = self.segmentation_pipeline(input_.volume)
            if self.return_fragments:
                return output_segm[1][final_crop], output_segm[0][final_crop]
            else:
                return output_segm[final_crop]

class BlockWiseSegmentationPipelineSolver(object):
    def __init__(self,
                 segmentation_pipeline,
                 crop_padding=False,
                 offsets=None,
                 nb_parallel_blocks=1,
                 nb_threads=8,
                 num_workers=1):
        """
        :param blockwise: if False, the whole dataset is processed together
        :param nb_parallel_blocks: how many blocks are solved in parallel
        :param nb_threads: nb threads used computations in every block
        :param num_workers: used to load affinities from file (probably not needed, since there is no augmentation)
        """
        self.segmentation_pipeline = segmentation_pipeline
        self.nb_offsets = len(offsets)

        if offsets is not None:
            assert len(offsets) == self.nb_offsets, "%i, %i" % (len(offsets), self.nb_offsets)
        self.offsets = offsets

        self.crop_padding = crop_padding

        self.nb_parallel_blocks = nb_parallel_blocks
        if nb_parallel_blocks != 1:
            raise NotImplementedError()
        # TODO: not necessary!
        self.nb_threads = nb_threads
        self.num_workers = num_workers


    @classmethod
    def from_config(cls, segmentation_pipeline, offsets, config):
        config = yaml2dict(config)
        crop_padding = config.get("crop_padding", False)
        nb_threads = config.get("nb_threads", 8)
        nb_parallel_blocks = config.get("nb_parallel_blocks", 1)
        num_workers = config.get("num_workers", 1)
        if offsets is not None:
            offsets = [tuple(off) for off in offsets]
        return cls(segmentation_pipeline,
                   crop_padding=crop_padding,
                   nb_threads=nb_threads,
                   nb_parallel_blocks=nb_parallel_blocks,
                   num_workers=num_workers,
                   offsets=offsets)


    def __call__(self, dataset):
        # build the output volume
        shape_affs = dataset.volume.shape
        assert shape_affs[0] == self.nb_offsets
        assert len(shape_affs) == 4
        shape_output = shape_affs[1:]

        output = np.ones(shape_output, dtype='int64') * (-1)

        # loader
        loader = SimpleParallelLoader(dataset, num_workers=self.num_workers)
        # mask to count the number of times a pixel was inferred

        max_label = 0
        while True:
            # TODO: parallelize (take care of the max label...)
            batches = loader.next_batch()
            if not batches:
                print("[*] All blocks were processed!")
                break

            assert len(batches) == 1
            assert len(batches[0]) == 2
            index, input_ = batches[0]
            print("[+] Processing block {} of {}.".format(index+1, len(dataset)))
            # print("[*] Input-shape {}".format(input_.shape))

            # get the slicings w.r.t. the current prediction and the output
            local_slicing, global_slicing = self.get_slicings(dataset.base_sequence[index],
                                                              input_.shape,
                                                              dataset.padding)
            # remove offset dim from slicing
            global_slicing = global_slicing[1:]
            local_slicing = local_slicing[1:]

            print("Global slice: {}".format(global_slicing))
            output_patch = self.segmentation_pipeline(input_)

            # TODO: ADD CROP OF PADDING. We should predict with a padding and then crop!

            # save predictions in the output
            output_patch = vigra.analysis.labelVolume(output_patch[local_slicing].astype(np.uint32))
            output[global_slicing] = output_patch + max_label
            max_label += output_patch.max() + 1

        # # crop padding from the outputs
        # crop = tuple(slice(pad[0], shape_output[i] - pad[1]) for i, pad in enumerate(dataset.padding[1:]))
        output[output==-1] = max_label + 1
        # return the prediction (not cropped)
        return output


    def get_slicings(self, slicing, shape, padding):
        # crop away the padding (we treat global as local padding) if specified
        # this is generally not necessary if we use blending
        if self.crop_padding:
            # slicing w.r.t the current output
            local_slicing = tuple(slice(pad[0], shape[i] - pad[1])
                                  for i, pad in enumerate(padding))
            # slicing w.r.t the global output
            global_slicing = tuple(slice(slicing[i].start + pad[0],
                                         slicing[i].stop - pad[1])
                                   for i, pad in enumerate(padding))
        # otherwise do not crop
        else:
            local_slicing = np.s_[:]
            global_slicing = slicing
        return local_slicing, global_slicing
