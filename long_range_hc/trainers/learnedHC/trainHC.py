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

from long_range_hc.datasets.segm_transform import ComputeStructuredWeightsWrongMerges

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
            self.segmToAffs_GT = Segmentation2AffinitiesFromOffsets(dim=3,
                                               offsets=self.options['HC_config']['offsets'],
                                               add_singleton_channel_dimension = True,
                                               retain_segmentation = True,
                                               use_gpu=True,
                                                                 boundary_erode_segmentation=[0,2,2]
            )

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
            self.segmToAffs_initSegm = Segmentation2AffinitiesFromOffsets(dim=3,
                                               offsets=[[-1, 0, 0], [0, -2, 0], [0, 0, -2]],
                                               add_singleton_channel_dimension = True,
                                               retain_segmentation = True,
                                               use_gpu=True,
                                                                 boundary_erode_segmentation=[0,1,1]
            )


        segm_tensor = segm_tensor.data

        affinities = []
        for b in range(segm_tensor.size()[0]):
            affinities.append(self.segmToAffs_initSegm(segm_tensor[b])[None, ...])
        affinities = torch.cat(affinities, dim=0)
        if not retain_segmentation:
            affinities = affinities[:, 1:]
        return affinities


    def compute_oversegm_loss_weights(self, output, target):
        """
        :param segm: [batch_size, z, x, y]
        """
        postproc_config = self.options['HC_config']
        offsets = postproc_config['offsets']
        nb_threads = postproc_config['nb_threads']

        if not hasattr(self, 'DTWS'):
            if postproc_config['postproc_type'] == 'DTWS':
                WSDT_kwargs = deepcopy(postproc_config.get('WSDT_kwargs', {}))
                self.postproc_alg = WatershedOnDistanceTransformFromAffinities(
                    offsets,
                    WSDT_kwargs.pop('threshold', 0.5),
                    WSDT_kwargs.pop('sigma_seeds', 0.),
                    invert_affinities=True,
                    return_hmap=False,
                    n_threads=nb_threads,
                    **WSDT_kwargs,
                    **postproc_config.get('prob_map_kwargs', {}))
            elif postproc_config['postproc_type'] == 'MWS':
                self.postproc_alg = DamWatershed(offsets,
                             min_segment_size=10,
                             invert_affinities=False,
                             n_threads=nb_threads,
                             **postproc_config.get('MWS_kwargs', {}))

            weighting = 0.0005 if 'weighting' not in postproc_config else postproc_config['weighting']
            self.get_loss_merge_weights = ComputeStructuredWeightsWrongMerges(
                                        offsets,
                                                dim=3,
                                                ignore_label=0,
                weighting=weighting,
                                                number_of_threads=nb_threads)

        is_cuda = output.is_cuda



        pool = ThreadPool()

        def compute_weights(b):
            segm = self.postproc_alg(output[b].cpu().detach().numpy())
            loss_weights_batch = self.get_loss_merge_weights(segm, target[b, 0].cpu().detach().numpy())
            # loss_weights_batch[0] = 1.
            # loss_weights_batch[3:] = 1.
            return from_numpy(loss_weights_batch[None, ...])

        # Parallelize computation of the weights:
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


    def apply_model(self, *inputs):
        if len(inputs) == 1:
            model_inputs = inputs[0]
        elif len(inputs) == 2:
            init_segm_vectors = inputs[1][:, 1:].float()
            binary_boundaries = self.computeSegmToAffsCUDA_initSegm(inputs[1][:, 0].float(), retain_segmentation=False)
            model_inputs = torch.cat([inputs[0], init_segm_vectors, binary_boundaries], dim=1)
        elif len(inputs) == 4:
            # offsets = self.options['HC_config']['offsets']
            initSegm_vectors = inputs[1][:,1:].float()
            finalSegm_vectors = inputs[2][:, 1:].float()
            underSegm_vectors = inputs[3][:, 1:].float()
            model_inputs = torch.cat([inputs[0], initSegm_vectors, finalSegm_vectors, underSegm_vectors], dim=1)
        elif len(inputs) == 3:
            # offsets = self.options['HC_config']['offsets']
            # initSegm_vectors = inputs[1][:,1:].float()
            finalSegm_vectors = inputs[2][:, 1:].float()
            model_inputs = torch.cat([inputs[0], finalSegm_vectors], dim=1)
        else:
            raise NotImplementedError()
        output = super(HierarchicalClusteringTrainer, self).apply_model(model_inputs)
        if isinstance(output, tuple):
            return output[0]
        else:
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

        loss_weights = None
            # Compute structured loss weights:
        if len(inputs) == 1:
            loss_weights = self.compute_model_0_weights(out_prediction, target)



        # # TODO: check if is a tuple and this works...
        # if isinstance(static_prediction, tuple):
        #     is_cuda = out_prediction.is_cuda
        # else:
        #     is_cuda = static_prediction.is_cuda
        if self.pre_train:
            if not validation:
                pass
                # TODO: update plots!
                self.plot_pretrain_batch({"raw":raw,
                                          "init_segm": init_segm_labels,
                                          "loss_weights": loss_weights,
                                          # "lookAhead1": inputs[2][:, 0],
                                          # "lookAhead2": inputs[3][:, 0],
                                          "stat_prediction":out_prediction,
                                          # "eroded_finalSegm":eroded_finalSegm,
                                          "target":target})
            else:
                if len(inputs) == 2 or len(inputs) == 4:
                    # Compute segmentation:
                    GT_labels = target.cpu().data.numpy()[:, 0]
                    pred_numpy = out_prediction.cpu().data.numpy()
                    init_segm_numpy = init_segm_labels.cpu().data.numpy()
                    segmentations = [self.postprocessing(pred,initSegm) for pred, initSegm in zip(pred_numpy, init_segm_numpy)]
                    try:
                        validation_scores = [cremi_score(gt, segm, return_all_scores=True) for gt, segm in
                                             zip(GT_labels, segmentations)]
                        validation_scores = [ [score[key] for key in score] for score in validation_scores]
                        print(validation_scores)
                    except ZeroDivisionError:
                        print("Error in computing scores...")
                        validation_scores = [ [0. , 0., 0., 0. ] for _ in pred_numpy]
                    self.criterion.validation_score = np.array(validation_scores).mean(axis=0)

                    var_segm = Variable(from_numpy(np.stack(segmentations)))
                    self.plot_pretrain_batch({"raw": raw,
                                              "stat_prediction": out_prediction,
                                              "init_segm": init_segm_labels,
                                              # "lookAhead1": inputs[2][:, 0],
                                              # "lookAhead2": inputs[3][:, 0],
                                              "target": target,
                                              "final_segm": var_segm,
                                              "GT_labels": target[:,0]},
                                             validation=True)




        # static_prediction = out_prediction

        # if len(inputs) == 3:
        #     loss = self.splitCNN_criterion(out_prediction, target)
        #
        #
        # else:
        loss = self.get_loss_static_prediction(out_prediction, target=target,
                                               validation=validation, loss_weights=loss_weights)

        print(loss.data.cpu().numpy())

        if validation and (len(inputs) == 3 or len(inputs) == 1) :
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
            # # Keep only largest prediction and invert (we want affinities):
            # static_prediction = 1. - out_prediction
            # # tick0 = time.time()
            #
            # self.model.set_static_prediction(False)
            #
            # # --------------------------
            # # Initialize LHC criterion:
            # # --------------------------
            # self.criterion.clear()
            # self.criterion.set_validation_mode(validation)
            # raw = inputs[0].cpu().data.numpy()
            # self.criterion.pass_batch_data_to_workers("set_raw_image", np.squeeze(raw, axis=1))
            #
            # if(backward or target is not None):
            #     # GT labels are in the first channel:
            #     GT_labels = target.cpu().data.numpy()[:,0]
            #     self.criterion.pass_batch_data_to_workers("set_targets", GT_labels)
            #
            # # Set initial segmentation based on static_prediction:
            # self.criterion.pass_batch_data_to_workers("set_static_prediction",
            #                                           static_prediction.cpu().data.numpy())
            #
            # # tick1 = time.time()
            # # print("Initialization: {} s".format(tick1 - tick0))
            #
            # # --------------------------
            # # Loop and perform mile-steps:
            # # --------------------------
            # dynamic_predictions = []
            # dynamic_loss_targets = []
            # dynamic_loss_weights = []
            # while not self.criterion.is_finished():
            #     # tick1 = time.time()
            #     # TODO: what if some workers are done...?
            #     dict_list, key_list = self.criterion.get_dynamic_inputs_milestep()
            #     all_dynamic_inputs = compose_model_inputs(dict_list, key_list, channel_axis=0)
            #
            #     if is_cuda:
            #         all_dynamic_inputs = [dyn_inp.cuda() for dyn_inp in all_dynamic_inputs]
            #
            #     # Keep static inputs (raw) on GPU:
            #     prediction_milestep = self.apply_model(*(inputs+all_dynamic_inputs))
            #
            #     # Check if dynamic prediction is multiscale:
            #     if isinstance(prediction_milestep, tuple):
            #         highRes_prediction = prediction_milestep[0]
            #     else:
            #         highRes_prediction = prediction_milestep
            #
            #     # Returned values are None in case of validation:
            #     milestep_loss_targets, milestep_loss_weights = self.criterion(highRes_prediction)
            #
            #     dynamic_predictions.append(prediction_milestep)
            #     dynamic_loss_targets.append(milestep_loss_targets)
            #     dynamic_loss_weights.append(milestep_loss_weights)
            #
            # if validation:
            #     self.criterion.run_clustering_on_pretrained_affs(start_from_pixels=False)
            #
            # # Plot data batch:
            # self.plot_batch(validation=not backward)
            #
            # if backward:
            #     for iter in range(len(dynamic_predictions)):
            #         # tick5 = time.time()
            #         dynamic_predictions_iter = dynamic_predictions[iter]
            #         dynamic_loss_targets_iter = dynamic_loss_targets[iter]
            #         dynamic_loss_weights_iter = dynamic_loss_weights[iter]
            #
            #         # dynamic_predictions_iter = torch.cat(dynamic_predictions, dim=0)
            #         # dynamic_loss_targets_iter = torch.cat(dynamic_loss_targets, dim=0)
            #         # dynamic_loss_weights_iter = torch.cat(dynamic_loss_weights, dim=0)
            #
            #         # print(dynamic_loss_targets_iter.size())
            #         # print(dynamic_predictions_iter.size())
            #
            #         # CLASSES TRAINING:
            #         # shape = dynamic_loss_targets.size()
            #         # dynamic_loss_targets = dynamic_loss_targets.view(shape[0], -1, shape[3], shape[4], shape[5])
            #         # dynamic_predictions = dynamic_predictions.view(shape[0], -1, shape[3], shape[4], shape[5])
            #
            #
            #         # # MY STRANGE IMPLEMENTATION WITH WEIGHTS (DOESN'T SEEM TO GIVE GREAT RESULTS)
            #         # print(dynamic_loss_targets.size())
            #         # merge_and_split_targets = dynamic_loss_targets[:,:,0:2]
            #         # self._structured_criterion.weight = self._compute_batch_weights(merge_and_split_targets).data
            #         # loss = loss + self._structured_criterion(dynamic_predictions, dynamic_loss_targets)
            #
            #         # dynamic_loss_weights = torch.cat(dynamic_loss_weights, dim=0)
            #
            #         # DICE LOSS TRAINING:
            #         # # Find number of trained pixels: (for batch average)
            #         # flat_weights = dynamic_loss_weights.view(-1)
            #         # zero_array = Variable(from_numpy(np.array([0.]))).cuda().float()
            #         # pixels_in_minibatch = torch.sum((flat_weights != zero_array).float())
            #         #
            #         # loss = loss.float()
            #         # loss = loss + self._structured_criterion(dynamic_predictions, dynamic_loss_targets, dynamic_loss_weights)/pixels_in_minibatch.clamp(min=1e-6)
            #
            #         # Inferno implementation of Dice Score:
            #         # Different classes are considered as separate channels:
            #         def compress_classes_and_offsets(tensor):
            #             size = tensor.size()
            #             return tensor.view(size[0], -1, size[3], size[4], size[5])
            #         dynamic_loss_targets_iter = compress_classes_and_offsets(dynamic_loss_targets_iter)
            #         dynamic_loss_weights_iter = compress_classes_and_offsets(dynamic_loss_weights_iter)
            #         if isinstance(dynamic_predictions_iter, (tuple, list)):
            #             # Multiscale predictions:
            #             dynamic_predictions_iter = tuple(compress_classes_and_offsets(pred) for pred in dynamic_predictions_iter)
            #         else:
            #             dynamic_predictions_iter = compress_classes_and_offsets(dynamic_predictions_iter)
            #         loss = self._structured_criterion(dynamic_predictions_iter, (dynamic_loss_targets_iter, dynamic_loss_weights_iter))
            #
            #         # tick6 = time.time()
            #         # print("Computing loss iter {}: {} s".format(iter, tick6 - tick5))
            #         #
            #         loss.backward()
            #         # break
            #
            #         # print("Backward iter {}: {} s".format(iter,time.time() - tick6))
            #
            #         # BCE:
            #         # flat_pred = dynamic_predictions.view(-1)
            #         # flat_targets = dynamic_loss_targets.view(-1)
            #         # flat_weights = dynamic_loss_weights.view(-1)
            #         #
            #         # BCE_criteria = BCELoss(weight=flat_weights, size_average=False)
            #         # zero_array = Variable(from_numpy(np.array([0.]))).cuda().float()
            #         #
            #         # pixels_in_minibatch = torch.sum((flat_weights != zero_array).float())
            #         # loss = loss.float()
            #         # loss = loss + BCE_criteria(flat_pred, flat_targets)/pixels_in_minibatch
            #
            #
            #
            # out_prediction = torch.cat(dynamic_predictions, dim=0)
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
            # TODO: change this shit... (multiply weights and sum directly in the loss...)
            if loss_weights is not None:
                loss_weights = loss_weights if not is_cuda else loss_weights.cuda()
                # if self.options['loss_type'] == 'soresen':
                #     sqrt_weights = torch.sqrt(loss_weights)
                #     prediction = prediction * sqrt_weights
                #     target[:,1:] = target[:,1:] * sqrt_weights
                loss = self.criterion.unstructured_loss(prediction, target)
                loss = loss.sum()

                # print("Soresen loss: ", loss.data.cpu().numpy())

                # TEMP for getting ignore label:
                loss_weights[loss == 0.] = 0.
                BCE_loss = self.BCE_loss(prediction, 1 - target[:,1:])
                BCE_loss = BCE_loss * loss_weights
                BCE_loss = BCE_loss.mean()

                # print("BCE loss: ", BCE_loss.data.cpu().numpy())

                loss = loss + BCE_loss * self.options['HC_config']['loss_BCE_factor']

                # if self.options['loss_type'] == 'BCE':
                #     loss = (loss * loss_weights)
                #
                # if 'loss_mul_factor' in self.options['HC_config']:
                #     factor = self.options['HC_config']['loss_mul_factor']
                #     factor = eval(factor) if isinstance(factor, str) else factor
                #     loss = loss * factor

                loss = loss.sum()


            else:
                loss = self.criterion.unstructured_loss(prediction, target)
                loss = loss.sum()
        if is_cuda:
            loss = loss.cuda()
        return loss

    # def get_loss(self, prediction, target=None, validation=False):
    #     if self.criterion.unstructured_loss is not None and not validation:
    #
    #         # make a validation ws step without calculating the expensive structured loss
    #         self.criterion.set_validation_mode(True)
    #         self.criterion(prediction)
    #         self.criterion.set_validation_mode(validation)
    #
    #         # TODO: replace with parameter
    #         nac = 2
    #         ndc = 12
    #         not_inverted_affinities = cat((prediction[:, :nac], 1-prediction[:, -ndc:]), 1)
    #         aff_target = cat((target[:, :nac+1], target[:, -ndc:]), 1).cuda()
    #         return self.criterion.unstructured_loss(not_inverted_affinities, aff_target)
    #     else:
    #         self.criterion.set_validation_mode(validation)
    #         return self.criterion(prediction).cuda()

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

