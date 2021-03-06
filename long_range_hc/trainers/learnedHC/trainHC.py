''' 
This module contains all training methods for the end to end hierarchical clustering training
'''
import numpy as np
from copy import deepcopy
import warnings
from multiprocessing.pool import ThreadPool

from torch import randn
from torch.nn.modules.loss import BCELoss
# from inferno.extensions.criteria.set_similarity_measures import GeneralizedDiceLoss
from long_range_hc.criteria.learned_HC.utils.temp_soresen_loss import SorensenDiceLossPixelWeights

from skunkworks.criteria.multi_scale_loss_weighted import MultiScaleLossMaxPoolWeighted
# FIXME: this has been changed!
# from skunkworks.postprocessing.pipelines import fixation_agglomerative_clustering_from_wsdt2d
from skunkworks.metrics.cremi_score import cremi_score

from inferno.extensions.criteria import SorensenDiceLoss

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from eta import ETA
import torch
from torch.autograd import Variable
from torch import from_numpy

from long_range_hc.postprocessing.segmentation_pipelines.agglomeration.fixation_clustering import \
    FixationAgglomeraterFromSuperpixels

from segmfriends.transform.inferno.temp_crap import ComputeStructuredWeightsWrongMerges

from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import InvertTarget, MaskTransitionToIgnoreLabel, RemoveSegmentationFromTarget
from long_range_hc.criteria.loss_transforms import ApplyMaskToBatch
from inferno.io.transform.base import Compose

from torch.nn.modules.loss import BCELoss

from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets

from skunkworks.inference.test_time_augmentation import TestTimeAugmenter
from skunkworks.inference.blending import Blending
from inferno.utils.io_utils import yaml2dict
from inferno.io.core import Zip
from skunkworks.inference.simple import SimpleParallelLoader
from skunkworks.postprocessing.watershed import DamWatershed

from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline


from skunkworks.postprocessing.watershed.wsdt import WatershedOnDistanceTransformFromAffinities


def compose_model_inputs(dictionary_list, key_list, channel_axis=-1):
    '''
    Helping function to compose CNN inputs based on the channel-keys defined in the config file.

        :param input_dictionary:    list of dictionaries containing the channels for each batch item
                                            [{'raw':array, 'affs':array, ..}, ...]
                                    and arrays should have the shape (z, x, y, channels).

        :param key_list: specify channels for every input:

                                    [ ['raw', 'affs', ...],
                                      ['lookAheads', 'keyB_input2', ...],
                                            ...,
                                    ]

        :return  list of pyTorch variables, e.g.
                 [input1, input2,...] where the shape now is (batch_size, channels, z, x, y)
    '''
    assert isinstance(dictionary_list, list)
    assert isinstance(key_list, list)
    num_inputs = len(key_list)
    batch_size = len(dictionary_list)

    assert channel_axis==-1 or channel_axis==0
    outputs = [None for _ in range(num_inputs)]

    for i in range(num_inputs):
        output_i = []
        for b in range(batch_size):
            channels = dictionary_list[b]
            output_i_b = []
            for key in key_list[i]:
                assert key in channels, "Key '{}' not found! Available channels: {}"\
                    .format(key, channels.keys())
                output_i_b.append(channels[key])
            output_i_b = np.concatenate(output_i_b, axis=channel_axis)
            ndim = output_i_b.ndim - 1
            # Move channel axis to first dim:
            if channel_axis==-1:
                output_i_b = np.transpose(output_i_b, (ndim,) + range(ndim))
            output_i.append(output_i_b)
        outputs[i] = Variable(from_numpy(np.stack(output_i, axis=0).astype(np.float32)))
    return outputs



class HierarchicalClusteringTrainer(Trainer):
    def __init__(self, model=None,
                 pre_train=True,
                 **trainer_kwargs):
        super(HierarchicalClusteringTrainer, self).__init__(model=model)

        assert pre_train, "Structure training is deprecated!"

        self.pre_train = pre_train
        self.options = trainer_kwargs



        if trainer_kwargs:
            self.BCE_loss = BCELoss(reduce=False)

            # LOOK-AHEAD criterion:
            self.lookahead_loss = SorensenDiceLoss()
            self.applyMask = ApplyMaskToBatch(targets_are_inverted=False)
            self.maskIgnoreLabel = MaskTransitionToIgnoreLabel(trainer_kwargs['HC_config']['offsets'],
                                                               targets_are_inverted=True)

            # self.splitCNN_criterion = LossWrapper(criterion=SorensenDiceLoss(),
            #                                 transforms=Compose(RemoveSegmentationFromTarget(),
            #                                                    InvertTarget()))


            # SETUP STRUCTURED CRITERION:
            # self._structured_criterion = SorensenDiceLossPixelWeights(channelwise=True)
            scaling_factor = trainer_kwargs['model_kwargs']['scale_factor']
            multiscale_loss_kwargs = trainer_kwargs.get('multiscale_loss_kwargs', {})

            if not trainer_kwargs['model_kwargs'].get('multiscale', False):
                multiscale_loss_kwargs['n_scales'] = 1
                multiscale_loss_kwargs.pop('scale_weights', None)
                scaling_factor = []

            self._structured_criterion = \
                MultiScaleLossMaxPoolWeighted(SorensenDiceLossPixelWeights(channelwise=True),
                                              scaling_factor,
                                              invert_target=False,
                                              targets_pool_mode='mean',
                                              weighted_loss=True,
                                              **multiscale_loss_kwargs)


            if self.pre_train:
                self.postprocessing = self.get_postprocessing(trainer_kwargs)

    def set_postproc_config(self, options):
        self.postproc_options = yaml2dict(options)

    def computeSegmToAffsCUDA_finalSegm(self, segm_tensor, retain_segmentation = True):
        """
        :param segm: [batch_size, z, x, y]
        """
        if not hasattr(self, 'segmToAffs_fnSegm'):
            self.segmToAffs_fnSegm = Segmentation2AffinitiesFromOffsets(dim=3,
                                               offsets=self.options['HC_config']['offsets'],
                                               add_singleton_channel_dimension = True,
                                               retain_segmentation = True,
                                               use_gpu=True,
                                                                 boundary_erode_segmentation=[0,2,2], # 9, 9
                                                                        return_eroded_labels=True
            )

        with torch.no_grad():
            segm_tensor = segm_tensor.data

            affinities = []
            for b in range(segm_tensor.size()[0]):
                affinities.append(self.segmToAffs_fnSegm(segm_tensor[b])[None, ...])
            affinities = torch.cat(affinities, dim=0)
            if not retain_segmentation:
                affinities = affinities[:, 1:]
        return affinities

    def computeSegmToAffsCUDA_GT(self, segm_tensor, retain_segmentation = True):
        """
        :param segm: [batch_size, z, x, y]
        """
        if not hasattr(self, 'segmToAffs_GT'):
            erode_boundary_thickness = self.options['HC_config']['erode_boundary_thickness'] if 'erode_boundary_thickness' in self.options['HC_config'] else 2
            boundary_erode_segmentation = [0,erode_boundary_thickness,erode_boundary_thickness] if erode_boundary_thickness != 0 else None
            self.segmToAffs_GT = Segmentation2AffinitiesFromOffsets(dim=3,
                                               offsets=self.options['HC_config']['offsets'],
                                               add_singleton_channel_dimension = True,
                                               retain_segmentation = True,
                                               use_gpu=True,
                                                                 boundary_erode_segmentation=boundary_erode_segmentation
            )

        with torch.no_grad():
            segm_tensor = segm_tensor.data

            affinities = []
            for b in range(segm_tensor.size()[0]):
                affinities.append(self.segmToAffs_GT(segm_tensor[b])[None, ...])
            affinities = torch.cat(affinities, dim=0)
            if not retain_segmentation:
                affinities = affinities[:, 1:]

        return affinities

    def computeSegmToAffsCUDA_initSegm(self, segm_tensor, retain_segmentation = True):
        """
        :param segm: [batch_size, z, x, y]
        """
        if not hasattr(self, 'segmToAffs_initSegm'):
            offsets = self.options['HC_config']['offsets']
            erode_boundary_thickness = self.options['HC_config'][
                    'erode_boundary_thickness'] if 'erode_boundary_thickness' in self.options['HC_config'] else 2
            boundary_erode_segmentation = [0, erode_boundary_thickness,
                                               erode_boundary_thickness] if erode_boundary_thickness != 0 else None
            # if include_long_range_affs:
            #     offsets += [[-4, 0, 0], [0, -12, 0], [0, 0, -12], [0, -27, 0], [0, 0, -27]]

            self.segmToAffs_initSegm = Segmentation2AffinitiesFromOffsets(dim=3,
                                               offsets=offsets,
                                               add_singleton_channel_dimension = True,
                                               retain_segmentation = True,
                                               use_gpu=True,
                                                                 boundary_erode_segmentation=boundary_erode_segmentation
            )

        with torch.no_grad():

            segm_tensor = segm_tensor.data

            affinities = []
            for b in range(segm_tensor.size()[0]):
                affinities.append(self.segmToAffs_initSegm(segm_tensor[b])[None, ...])
            affinities = torch.cat(affinities, dim=0)
            if not retain_segmentation:
                affinities = affinities[:, 1:]
        return affinities


    def computeSegmToAffsCUDA_initSegm_shortRange(self, segm_tensor, retain_segmentation = True):
        """
        :param segm: [batch_size, z, x, y]
        """
        if not hasattr(self, 'segmToAffs_initSegm_shortRange'):
            offsets = [[-1, 0, 0], [0, -2, 0], [0, 0, -2]]
            boundary_erode_segmentation = [0, 2, 2]
            self.segmToAffs_initSegm_shortRange = Segmentation2AffinitiesFromOffsets(dim=3,
                                               offsets=offsets,
                                               add_singleton_channel_dimension = True,
                                               retain_segmentation = True,
                                               use_gpu=True,
                                                                 boundary_erode_segmentation=boundary_erode_segmentation
            )

        with torch.no_grad():

            segm_tensor = segm_tensor.data

            affinities = []
            for b in range(segm_tensor.size()[0]):
                affinities.append(self.segmToAffs_initSegm_shortRange(segm_tensor[b])[None, ...])
            affinities = torch.cat(affinities, dim=0)
            if not retain_segmentation:
                affinities = affinities[:, 1:]
        return affinities


    def compute_oversegm_loss_weights(self, output, target, init_segm=None):
        """
        :param segm: [batch_size, z, x, y]

        If the weights should not be computed, the function returns a None object
        """
        options = self.options['HC_config']
        offsets = options['offsets']
        nb_threads = options['nb_threads']

        struct_weights_kwargs = options.get('struct_weights_kwargs', {})

        # Check if loss weights should be computed:
        if struct_weights_kwargs.get('trained_mistakes', None) not in ['all_mistakes', 'only_merge_mistakes',
                                                             'only_split_mistakes']:
            return None

        if not hasattr(self, 'segmentation_pipeline'):
            post_proc_config = self.postproc_options
            if init_segm is not None:
                assert post_proc_config['start_from_given_segm'], 'Init. segm. is given. Please update postproc.'
            else:
                assert not post_proc_config['start_from_given_segm'], 'Init. segm. is NOT given. Please update postproc.'

            post_proc_config.pop('return_fragments', False)
            post_proc_config.pop('nb_threads')
            invert_affinities = post_proc_config.pop('invert_affinities', False)
            segm_pipeline_type = post_proc_config.pop('segm_pipeline_type', 'gen_HC')

            self.segmentation_pipeline = get_segmentation_pipeline(
                segm_pipeline_type,
                offsets,
                nb_threads=nb_threads,
                invert_affinities=invert_affinities,
                return_fragments=False,
                **post_proc_config
            )

            self.get_loss_merge_weights = ComputeStructuredWeightsWrongMerges(
                                        offsets,
                                                dim=3,
                                                ignore_label=0,
                                                number_of_threads=nb_threads,
                                                **struct_weights_kwargs)


        is_cuda = output.is_cuda

        pool = ThreadPool()

        def compute_weights(b):
            if init_segm is None:
                segm = self.segmentation_pipeline(output[b].cpu().detach().numpy())
            else:
                segm = self.segmentation_pipeline(output[b].cpu().detach().numpy(), init_segm[b].cpu().detach().numpy())
            loss_weights_batch = self.get_loss_merge_weights(segm, target[b, 0].cpu().detach().numpy())
            # loss_weights_batch[0] = 1.
            # loss_weights_batch[3:] = 1.
            return from_numpy(loss_weights_batch[None, ...])


        # Parallelize computation of the weights for different batches:
        loss_weights = pool.map(compute_weights,
                 range(output.size()[0]))

        pool.close()
        pool.join()

        # for b in range(output.size()[0]):
        #     segm = self.DTWS(output[b].cpu().numpy())
        #     loss_weights_batch = self.get_loss_merge_weights(segm, target[b,0].cpu().numpy())
        #     loss_weights.append(from_numpy(loss_weights_batch[None, ...]))

        loss_weights = torch.cat(loss_weights, dim=0)
        loss_weights = loss_weights.cuda() if is_cuda else loss_weights
        return loss_weights

    def compute_model_0_weights(self, output, target):
        """
        :param segm: [batch_size, z, x, y]
        """
        boundaries = target[:, 1:] == 0
        loss_weights = torch.zeros(size=boundaries.size(), requires_grad=False).cuda()
        loss_weights[boundaries] = 1.
        # Focus only on local offsets:
        loss_weights[:, 0] = 0.
        loss_weights[:, 3:] = 0.
        return loss_weights


    def get_postprocessing(self, options):
        postproc_config = options['HC_config']
        affinity_offsets = postproc_config['offsets']
        nb_threads = postproc_config['nb_threads']
        return FixationAgglomeraterFromSuperpixels(
            affinity_offsets,
            n_threads=nb_threads,
            invert_affinities=postproc_config.get('invert_affinities', False),
            **postproc_config['agglomeration_kwargs']
        )

    def __getstate__(self):
        state_dict = dict(self.__dict__)
        state_dict.pop('postprocessing', None)

        print("CIao")
        return state_dict

    def __setstate__(self, state_dict):
        state_dict['postprocessing'] = self.get_postprocessing(state_dict['options'])
        print("Restoring!!")
        self.__dict__.update(state_dict)

    def register_visualization_callback(self, plotter):
        self.register_callback(plotter)
        self.plotter = plotter

    def register_unstructured_criterion(self, us_crit):
        self.criterion.unstructured_loss = us_crit

    def plot_batch(self, batch_list=None, validation=False):
        if self.plotter is not None and (self.plotter.check_plot() or validation):
            image_data = self.criterion.get_plot_images()
            suffix = '' if not validation else 'valid'
            self.plotter.enqueue_batch_plots(image_data, batch_list, suffix)

    def plot_pretrain_batch(self, img_data, batch=0, validation=False):
        if self.plotter is not None and (self.plotter.check_plot() or validation):
            suffix = '' if not validation else 'valid'
            self.plotter.plot_batch_pretraining(img_data, batch, suffix)

    def map_labels_to_embedding_space(self, input_):
        """
        :param input_: Numpy array with expected shape: (1 batch, 1 channel, 5, 324, 324)
        """
        pass


    def apply_model(self, *inputs):
        rnd_ints = None
        boundary_segmentation_mask = None
        if len(inputs) == 1:
            model_inputs = inputs[0]
        elif len(inputs) == 2:

            if self.options['model_type'] == 'splitCNN':
                assert inputs[1].size(1) != 1, "SplitCNN requires embedding vectors"
                init_segm_vectors = inputs[1][:, 1:].float()
                init_segm_vectors = torch.zeros_like(init_segm_vectors)
                binary_boundaries = self.computeSegmToAffsCUDA_initSegm_shortRange(inputs[1][:, 0].float(),
                                                                        retain_segmentation=False)
                binary_boundaries = torch.zeros_like(binary_boundaries)
                model_inputs = torch.cat([inputs[0], init_segm_vectors, binary_boundaries], dim=1)
            elif self.options['model_type'] == 'mergeCNN':
                # assert inputs[1].size(1) == 1, "MergeCNN does not use embedding vectors"
                binary_boundaries = self.computeSegmToAffsCUDA_initSegm(inputs[1][:, 0].float(),
                                                                        retain_segmentation=False)
                boundary_segmentation_mask = binary_boundaries
                model_inputs = torch.cat([inputs[0], binary_boundaries], dim=1)
            else:
                raise NotImplementedError()

        elif len(inputs) == 4:
            def get_batch_inputs(b):
                batch_inputs = [inp[[b],...] for inp in inputs]
                # Throw a dice and decide how many inputs to keep (for every batch):
                rnd_int = np.random.randint(0, 3)
                if rnd_int == 0:
                    print("R:", end=' ')
                    batch_inputs = batch_inputs[0]
                else:
                    initSegm_vectors = batch_inputs[1][:, 1:].float()
                    # Init segm. is always 2D, do not pass mask along z:
                    binary_bound_initSegm = self.computeSegmToAffsCUDA_initSegm(batch_inputs[1][:, 0].float(),
                                                                            retain_segmentation=False)[:,1:]
                    if rnd_int == 1:
                        print("R+iSegm:", end=' ')
                        batch_inputs = torch.cat([batch_inputs[0], initSegm_vectors, binary_bound_initSegm], dim=1)
                    else:
                        assert rnd_int == 2
                        print("R+iSegm+lAhead:" , end=' ')
                        finalSegm_vectors = batch_inputs[2][:, 1:].float()
                        binary_bound_finalSegm = self.computeSegmToAffsCUDA_initSegm(batch_inputs[2][:, 0].float(),
                                                                                    retain_segmentation=False)
                        underSegm_vectors = batch_inputs[3][:, 1:].float()
                        binary_bound_underSegm = self.computeSegmToAffsCUDA_initSegm(batch_inputs[3][:, 0].float(),
                                                                                 retain_segmentation=False)
                        batch_inputs = torch.cat([batch_inputs[0], initSegm_vectors, binary_bound_initSegm,
                                                  finalSegm_vectors, binary_bound_finalSegm,
                                                  underSegm_vectors, binary_bound_underSegm], dim=1)

                # Fill missing channels:
                nb_channels = self.options['pretrained_model_kwargs']['in_channels']
                missing_channels = nb_channels - batch_inputs.size(1)
                if missing_channels != 0:
                    missing_shape = (1, missing_channels) + batch_inputs.size()[2:]
                    batch_inputs = torch.cat([batch_inputs, torch.zeros(*missing_shape).cuda()], dim=1)
                return batch_inputs, rnd_int

            btch_size = inputs[0].size(0)
            pool = ThreadPool(processes=3)
            batch_inputs, rnd_ints = zip(*pool.map(get_batch_inputs, range(btch_size)))
            pool.close()
            pool.join()
            model_inputs = torch.cat(batch_inputs, dim=0)
            rnd_ints = from_numpy(np.expand_dims(np.array(rnd_ints), axis=1))

        elif len(inputs) == 3:
            # offsets = self.options['HC_config']['offsets']
            # initSegm_vectors = inputs[1][:,1:].float()
            finalSegm_vectors = inputs[2][:, 1:].float()
            model_inputs = torch.cat([inputs[0], finalSegm_vectors], dim=1)
        else:
            raise NotImplementedError()
        output = super(HierarchicalClusteringTrainer, self).apply_model(model_inputs)
        output = output[0] if isinstance(output, tuple) else output

        # if rnd_ints is not None:
        #     output = (output, rnd_ints)
        if boundary_segmentation_mask is not None:
            output = (output, boundary_segmentation_mask)

        print(output.data.cpu().numpy().mean())
        return output


    def apply_model_and_loss(self, inputs, target, backward=True, mode=None):
        """
        :type  inputs: list
        :param inputs: list of inputs, each with shape (batch_size, channels, z, x, y) on cuda
                        - atm inputs has only one channel with the raw image

        :type target: Variable
        :param target: Variable of shape (channels, batch_size, z, x, y) NOT on cuda
                        - first channel is label image
                        - following channels are affinities labels

        Affinities computed during the pre-processing are REAL affinities (1 = merge, 0 = split)

        """
        assert self.pre_train, "Structured training is deprecated!"

        # print(inputs[0][:,0].cpu().data.numpy().shape)
        validation = not backward
        # Check because for some reason it does not expect batch axis...?
        # inputs = [inputs[0], inputs[1]]

        # Compute target affinities on GPU:
        target = self.computeSegmToAffsCUDA_GT(target[:,0].cuda())

        eroded_finalSegm = None
        if len(inputs) == 3:
            # Targets for split-CNN:
            gt_labels = target[:,[0]].data
            finalSegmTensor = inputs[2][:, 0].data.clone() + 2
            finalSegmTensor[target[:, 0].data.long() == 0] = 0
            finalSegmTensor[target[:, 0].data.long() == 1] = 1

            finalSegm_affs = self.computeSegmToAffsCUDA_finalSegm(finalSegmTensor)
            eroded_finalSegm = Variable(finalSegm_affs[:, 0], requires_grad=False)

            # Only when I have boundary on GT (0) and not on the segmentation (1)
            split_targets_affs = 1 - ((finalSegm_affs[:, 1:] == 1) * (target[:, 1:].data == 0))
            target = Variable(torch.cat([gt_labels, split_targets_affs.float()],dim=1))

        # Combine raw and oversegmentation:
        raw = inputs[0][:,0]

        init_segm_labels = inputs[1][:, 0] if len(inputs) > 1 else None
        # if len(inputs) == 2:
        #     pass
        # elif len(inputs) == 4:
        #     init_segm_labels = inputs[1][:, 0]
        # else:
        #     raise NotImplementedError()


        if self.pre_train:
            # Legacy:
            if hasattr(self.model, 'set_pre_train_mode'):
                self.model.set_pre_train_mode(True)
            out_prediction = self.apply_model(*inputs)
        else:
            raise DeprecationWarning()
            # self.model.set_static_prediction(True)
            # static_prediction = self.apply_model(*inputs)

        boundary_segmentation_mask = None
        if isinstance(out_prediction, tuple):
            boundary_segmentation_mask = out_prediction[1]
            out_prediction = out_prediction[0]

        # # Compute structured loss weights:
        # loss_weights = self.compute_oversegm_loss_weights(out_prediction, target,
        #                                                       init_segm=init_segm_labels)


        # Modify targets:
        if len(inputs) == 2:

            # TODO: check this:
            # Boundary masks are affinities (boundaries: 0, merge: 1)
            loss_weights = torch.ones_like(target[:, 1:])
            if self.options['model_type'] == 'splitCNN':
                boundary_segmentation_mask = self.computeSegmToAffsCUDA_initSegm(inputs[1][:, 0].float(),
                                                                        retain_segmentation=False)

                # Here we should not train boundaries that should be merged in future:
                # print(boundary_segmentation_mask.size(), target[:, 1:].size())
                loss_weights[(boundary_segmentation_mask == 0.) * (target[:, 1:] == 1.)] = 0

                if 'modify_targets' in self.options:
                    if self.options['modify_targets']:
                        # MODIFY TARGETS:
                        new_targets = torch.ones_like(target)
                        new_targets[:,0] = target[:,0]
                        new_targets[:, 1:][(target[:, 1:] == 0.) * (boundary_segmentation_mask == 1.)] = 0.
                        target = new_targets

            elif self.options['model_type'] == 'mergeCNN':
                # Here we should not train boundaries that were previously wrongly merged:
                loss_weights[(boundary_segmentation_mask == 1.) * (target[:,1:] == 0.)] = 0

                if 'modify_targets' in self.options:
                    if self.options['modify_targets']:
                        # MODIFY TARGETS:
                        new_targets = torch.ones_like(target)
                        new_targets[:, 0] = target[:, 0]
                        new_targets[:,1:][(target[:,1:] == 1.) * (boundary_segmentation_mask == 0.)] = 0.
                        target = new_targets

                #         target[:,2:4] = 1 - target[:,2:4]
                #         target[:,8:] = 1 - target[:,8:]




        # # TODO: check if is a tuple and this works...
        # if isinstance(static_prediction, tuple):
        #     is_cuda = out_prediction.is_cuda
        # else:
        #     is_cuda = static_prediction.is_cuda





        # static_prediction = out_prediction

        # if len(inputs) == 3:
        #     loss = self.splitCNN_criterion(out_prediction, target)
        #
        #
        # else:
        loss, loss_weights = self.get_loss_static_prediction(out_prediction, target=target,
                                               validation=validation, loss_weights=loss_weights)

        print(loss.data.cpu().numpy())

        # if 'invert_xy_targets' in self.options:
        #     if self.options['invert_xy_targets']:
        #         out_prediction[:, 1:3] = 1 - out_prediction[:, 1:3]
        #         out_prediction[:, 7:] = 1 - out_prediction[:, 7:]

        if self.pre_train:
            if not validation:
                pass
                # TODO: update plots!
                self.plot_pretrain_batch({"raw":raw,
                                          "init_segm": init_segm_labels,
                                          "loss_weights": loss_weights,
                                          # "rnd_ints": rnd_ints,
                                          # "lookAhead1": inputs[2][:, 0],
                                          # "lookAhead2": inputs[3][:, 0],
                                          "stat_prediction":out_prediction,
                                          # "eroded_finalSegm":eroded_finalSegm,
                                          "target":target})
            # else:
            #     if len(inputs) == 4:
            #         # Compute segmentation:
            #         GT_labels = target.cpu().data.numpy()[:, 0]
            #         pred_numpy = out_prediction.cpu().data.numpy()
            #         init_segm_numpy = init_segm_labels.cpu().data.numpy()
            #         segmentations = [self.postprocessing(pred,initSegm) for pred, initSegm in zip(pred_numpy, init_segm_numpy)]
            #         try:
            #             validation_scores = [cremi_score(gt, segm, return_all_scores=True) for gt, segm in
            #                                  zip(GT_labels, segmentations)]
            #             validation_scores = [ [score[key] for key in score] for score in validation_scores]
            #             print(validation_scores)
            #         except ZeroDivisionError:
            #             print("Error in computing scores...")
            #             validation_scores = [ [0. , 0., 0., 0. ] for _ in pred_numpy]
            #         self.criterion.validation_score = np.array(validation_scores).mean(axis=0)
            #
            #         var_segm = Variable(from_numpy(np.stack(segmentations)))
            #         self.plot_pretrain_batch({"raw": raw,
            #                                   "stat_prediction": out_prediction,
            #                                   "init_segm": init_segm_labels,
            #                                   # "lookAhead1": inputs[2][:, 0],
            #                                   # "lookAhead2": inputs[3][:, 0],
            #                                   "target": target,
            #                                   "final_segm": var_segm,
            #                                   "GT_labels": target[:,0]},
            #                                  validation=True)


        # if validation and (len(inputs) == 3 or len(inputs) == 1 or len(inputs) == 2) :
        self.criterion.validation_score = [loss.cpu().data.numpy()]


            # if len(inputs) == 4:
            #     # Compute look-ahead additional loss:
            #
            #     finalSegm_affs = 1 - self.computeSegmToAffsCUDA(inputs[2][:, 0].cuda(), retain_segmentation=False)
            #
            #     finalSegm_affs_masked, target_masked = self.maskIgnoreLabel(finalSegm_affs, target)
            #     target_affs_masked = 1 - target_masked[:,1:]
            #     mask = target_affs_masked != finalSegm_affs_masked
            #
            #
            #     out_prediction_masked, target_affs_masked = self.applyMask(out_prediction, target_affs_masked, mask)
            #     loss += 3 * self.lookahead_loss(out_prediction_masked, target_affs_masked)
            #
            #     underSegm_affs = 1 - self.computeSegmToAffsCUDA(inputs[3][:, 0].cuda(), retain_segmentation=False)
            #
            #     underSegm_affs_masked, target_masked = self.maskIgnoreLabel(underSegm_affs, target)
            #     target_affs_masked = 1 - target_masked[:, 1:]
            #     mask = target_affs_masked != underSegm_affs_masked
            #
            #     out_prediction_masked, target_affs_masked = self.applyMask(out_prediction, target_affs_masked, mask)
            #     loss += 2 * self.lookahead_loss(out_prediction_masked, target_affs_masked)
            #
            #     # from matplotlib import pyplot as plt
            #     # DEF_INTERP = 'none'
            #     # f, ax = plt.subplots(ncols=2, nrows=2,
            #     #                      figsize=(2, 2))
            #     # ax[0, 0].matshow(target_affs_masked[0,4,2].cpu().data.numpy(), cmap='gray', alpha=1., interpolation=DEF_INTERP)
            #     # ax[1, 0].matshow(finalSegm_affs_masked[0,4,2].cpu().data.numpy(), cmap='Reds', alpha=1., interpolation=DEF_INTERP)
            #     # ax[1, 1].matshow(mask[0,4,2].cpu().data.numpy(), cmap='Greens', alpha=1., interpolation=DEF_INTERP)
            #     # plt.subplots_adjust(wspace=0, hspace=0)
            #     # f.savefig('/net/hciserver03/storage/abailoni/learnedHC/input_segm/debug_test/debug_plots.pdf', format='pdf')
            #     # plt.clf()
            #     # plt.close('all')
            #     #




        # print("Loss: {}".format(loss.data.cpu().numpy()))



        if not self.pre_train:
            raise DeprecationWarning()
        else:
            # Backprop
            if backward:
                loss.backward()



        # TODO: out_prediction will be used to evaluate a possible metric
        # but it is possible to hack this and make the metric communicate with the criterion,
        # so other dynamic LHC data could be shared in this way..
        return out_prediction, loss

    def get_loss_static_prediction(self, prediction, target=None, validation=False, loss_weights=None):
        if isinstance(prediction, tuple):
            is_cuda = prediction[0].is_cuda
        else:
            is_cuda = prediction.is_cuda
        if not self.pre_train or target is None:
            loss = Variable(from_numpy(np.array([0.], dtype=np.float)))
        else:
            target = target if not is_cuda else target.cuda()
            loss = self.criterion.unstructured_loss(prediction, target)

            if loss_weights is not None:
                loss = loss * loss_weights
                # TODO: change this shit... (multiply weights and sum directly in the loss...)
                # loss_weights = loss_weights if not is_cuda else loss_weights.cuda()
                # # if self.options['loss_type'] == 'soresen':
                # #     sqrt_weights = torch.sqrt(loss_weights)
                # #     prediction = prediction * sqrt_weights
                # #     target[:,1:] = target[:,1:] * sqrt_weights
                # loss = self.criterion.unstructured_loss(prediction, target)
                # # TEMP for getting ignore label:
                # # loss_weights[loss == 0.0] = 0.
                #
                # loss = loss.sum()
                #
                # # print("Soresen loss: ", loss.data.cpu().numpy())
                #
                #
                # BCE_loss = self.BCE_loss(prediction, 1 - target[:,1:])
                # BCE_loss = BCE_loss * loss_weights
                # BCE_loss = BCE_loss.mean()
                #
                # # print("BCE loss: ", BCE_loss.data.cpu().numpy())
                #
                # loss = loss + BCE_loss * self.options['HC_config']['loss_BCE_factor']
                #
                # # if self.options['loss_type'] == 'BCE':
                # #     loss = (loss * loss_weights)
                # #
                # # if 'loss_mul_factor' in self.options['HC_config']:
                # #     factor = self.options['HC_config']['loss_mul_factor']
                # #     factor = eval(factor) if isinstance(factor, str) else factor
                # #     loss = loss * factor
                #
                # loss = loss.sum()


            loss = loss.sum()
        if is_cuda:
            loss = loss.cuda()
        return loss, loss_weights

    def to_device(self, objects):
        if isinstance(objects, (list, tuple)):
            transfered_objects = [self.to_device(_object) if i < len(objects)-1
                              else _object for i, _object in enumerate(objects)]
            return type(objects)(transfered_objects)
        else:
            return objects.cuda() if self._use_cuda else objects


    # ---------
    # TEMPORARY INFERENCE METHODS (should be moved to the more basic class)
    # ---------
    def build_infer_engine(self,
                           infer_config):

        # GET CONFIG DATA: ---------------
        config = yaml2dict(infer_config)
        gpu = config.get("gpu", 0)

        blending_config = config.get("blending_config", None)
        blending = None if blending_config is None else \
            Blending(**blending_config)

        # For now we only support loading default TDAs from config
        augmentation_config = config.get("augmentation_config", None)
        augmenter = None if augmentation_config is None else \
            TestTimeAugmenter.default_tda(**augmentation_config)

        crop_padding = config.get("crop_padding", False)
        num_workers = config.get("num_workers", 8)
        offsets = config.get("offsets", None)
        if offsets is not None:
            offsets = [tuple(off) for off in offsets]

        # BUILD INFERENCE ENGINE: ---------------
        # FIXME FIXME
        self.out_channels = len(offsets)

        if offsets is not None:
            assert len(offsets) == self.out_channels, "%i, %i" % (len(offsets), self.out_channels)
        self.channel_offsets = offsets

        # TODO validate gpu
        self.gpu = gpu

        if blending is not None:
            assert isinstance(blending, Blending)
        self.blending = blending

        if augmenter is not None:
            assert isinstance(augmenter, TestTimeAugmenter)
        self.augmenter = augmenter

        self.crop_padding = crop_padding
        self.num_workers = num_workers


    def get_slicings(self, slicing, shape, padding):
        # crop away the padding (we treat global as local padding) if specified
        # this is generally not necessary if we use blending
        if self.crop_padding:
            assert  all([slicing[i].step == 1 for i in range(len(padding))]), "Downscaling option not implemented yet!"
            # slicing w.r.t the current output
            local_slicing = tuple(slice(pad[0],
                                        shape[i] - pad[1])
                                  for i, pad in enumerate(padding))
            # slicing w.r.t the global output
            global_slicing = tuple(slice(slicing[i].start + pad[0],
                                         slicing[i].stop - pad[1])
                                   for i, pad in enumerate(padding))
        # otherwise do not crop
        else:
            local_slicing = tuple(slice(None, None)
                                  for i, pad in enumerate(padding))
            global_slicing = slicing
        return local_slicing, global_slicing

    def wrap_numpy_in_variable(self, *numpy_arrays_,
                               requires_grad=False,
                               volatile=True
                               ):
        variables = []
        for np_input in numpy_arrays_:
            # Add batch and channel axes if not present:
            for _ in range(5 - np_input.ndim):
                np_input = np.expand_dims(np_input, axis=0)
            tensor = torch.from_numpy(np_input).cuda(self.gpu).float()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                variables.append(
                    Variable(tensor,
                             requires_grad=requires_grad,
                             volatile=volatile)
                )
        return variables

    def unwrap_prediction(self, output):
        # FIXME: fix this mess that does not work if batch is different from 1!
        if isinstance(output, tuple):
            output = output[0]
        out = output.cpu().data.numpy()
        assert out.shape[0] == 1
        out = np.squeeze(out)
        return out

    def infer_patch(self, *inputs_):
        with torch.no_grad():
            # infer with or without TDA
            if self.augmenter is None:
                # without TDA
                vars = self.wrap_numpy_in_variable(*inputs_,
                                                   requires_grad=False,
                                                   volatile=True)
                output = self.apply_model(*vars)
            else:
                raise DeprecationWarning()
                # with TDA
                # output = self.augmenter(input_,
                #                         partial(self.apply_model_infer, predict_images=predict_images),
                #                         offsets=self.channel_offsets)
            output = self.unwrap_prediction(output)
        return output

    def infer(self, dataset):
        dataset_raw = dataset.raw_volume if isinstance(dataset, Zip) else dataset

        # build the output volume
        shape = dataset_raw.volume.shape

        print("Whole shape to predict: {}".format(shape))
        output = np.zeros((self.out_channels,) + shape, dtype='float32')
        # loader
        loader = SimpleParallelLoader(dataset, num_workers=self.num_workers,
                                      maxsize_queue=3)
        # mask to count the number of times a pixel was infered
        mask = np.zeros(shape)

        # do the actual inference
        # TODO verbosity and logging
        while True:
            batches = loader.next_batch()
            if not batches:
                print("[*] Inference finished")
                break

            assert len(batches) == 1
            assert len(batches[0]) == 2
            index, inputs_ = batches[0]
            inputs_ = inputs_ if isinstance(inputs_, list) else [inputs_]
            print("[+] Inferring batch {} of {}.".format(index, len(dataset)))
            # print("[*] Input-shape {}".format(inputs_.shape))

            # get the slicings w.r.t. the current prediction and the output
            local_slicing, global_slicing = self.get_slicings(dataset_raw.base_sequence[index],
                                                              inputs_[0].shape,
                                                              dataset_raw.padding)
            print("Global slice: {}".format(global_slicing))
            output_patch = self.infer_patch(*inputs_)

            if self.blending is not None:
                output_patch, blending_mask = self.blending(output_patch)
                mask[global_slicing] += blending_mask[local_slicing]
            else:
                mask[global_slicing] += 1
            # add slicing for the channel
            global_slicing = (slice(None),) + global_slicing
            local_slicing = (slice(None),) + local_slicing
            # add up predictions in the output
            output[global_slicing] += output_patch[local_slicing]

        # # crop padding from the outputs
        # # This is only used to crop the dataset in the end
        ds_ratio = dataset_raw.downsampling_ratio

        crop = tuple(slice(pad[0],
                           shape[i] - pad[1],
                           ds_ratio[i]) for i, pad in enumerate(
            dataset_raw.padding))
        out_crop = (slice(None),) + crop
        output = output[out_crop]
        mask = mask[crop]

        # divide by the mask to normalize all pixels
        assert (mask != 0).all()
        assert mask.shape == output.shape[1:]
        output /= mask

        # return the prediction
        return output

class ParameterNoiseCallback(Callback):
    def __init__(self, options):
        super(ParameterNoiseCallback, self).__init__()
        self.options = options

    def begin_of_training_iteration(self, **_):
        # add gaussian noise to all parameters
        self.added_noise = []
        for param in self.trainer.model.parameters():
            noise = randn(param.size())*self.options.noise_sigma
            noise = noise.cuda()
            self.added_noise.append(noise)
            param.data += noise

    def end_of_training_iteration(self, **_):
        for param, noise in zip(self.trainer.model.parameters(), self.added_noise):
            param.data -= noise

class AlphaAnnealingCallback(Callback):
    def __init__(self, start, factor, step):
        super(AlphaAnnealingCallback, self).__init__()
        self.alpha_set = start
        self.factor = factor
        self.step = step

    def begin_of_training_iteration(self, **_):
        if self.trainer._iteration_count % self.step == 0:
            if abs(self.alpha_set) < 16:
                self.alpha_set *= self.factor
                print("setting alpha to ", self.alpha_set)
                self.trainer.criterion.set_alpha(self.alpha_set)

class ETACallback(Callback):
    def __init__(self, max_epoch):
        super(ETACallback, self).__init__()
        self.eta_estimator = ETA(max_epoch)

    def end_of_training_iteration(self, **_):
        self.eta_estimator.print_status()

class SaveModelCallback(Callback):
    def __init__(self, save_every):
        super(Callback, self).__init__()
        self.save_every = save_every

    def end_of_training_iteration(self, **_):
        if self.trainer._iteration_count % self.save_every == 0:
            self.trainer.save_model()

