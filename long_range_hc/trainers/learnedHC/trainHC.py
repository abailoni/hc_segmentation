''' 
This module contains all training methods for the end to end hierarchical clustering training
'''
import numpy as np
import random
import time
import h5py

from torch import randn
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

from torch.nn.modules.loss import BCELoss

from skunkworks.inference.test_time_augmentation import TestTimeAugmenter
from skunkworks.inference.blending import Blending
from inferno.utils.io_utils import yaml2dict
from inferno.io.core import Zip
from skunkworks.inference.simple import SimpleParallelLoader


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
        assert len(inputs) == 2
        init_segm_vectors = inputs[1][:, 1:]
        model_inputs = torch.cat([inputs[0], init_segm_vectors], dim=1)
        output = super(HierarchicalClusteringTrainer, self).apply_model(model_inputs)
        return output[0]



    def apply_model_and_loss(self, inputs, target, backward=True):
        """
        :type  inputs: list
        :param inputs: list of inputs, each with shape (batch_size, channels, z, x, y) on cuda
                        - atm inputs has only one channel with the raw image

        :type target: Variable
        :param target: Variable of shape (channels, batch_size, z, x, y) NOT on cuda
                        - first channel is label image
                        - following channels are affinities labels

        """
        assert self.pre_train, "Structured training is deprecated!"

        # print(inputs[0][:,0].cpu().data.numpy().shape)
        validation = not backward
        # Check because for some reason it does not expect batch axis...?

        # Combine raw and oversegmentation:
        raw = inputs[0][:,0]
        init_segm_labels = inputs[1][:,0]


        if self.pre_train:
            self.model.set_pre_train_mode(True)
            out_prediction = self.apply_model(*inputs)
        else:
            raise DeprecationWarning()
            # self.model.set_static_prediction(True)
            # static_prediction = self.apply_model(*inputs)



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
                                          "init_segm":init_segm_labels,
                                          "stat_prediction":out_prediction,
                                          "target":target})
            else:
                pass
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
                                          "target": target,
                                          "final_segm": var_segm,
                                          "GT_labels": target[:,0]},
                                         validation=True)




        # static_prediction = out_prediction

        loss = self.get_loss_static_prediction(out_prediction, target=target,
                                               validation=validation)




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

    def get_loss_static_prediction(self, prediction, target=None, validation=False):
        if isinstance(prediction, tuple):
            is_cuda = prediction[0].is_cuda
        else:
            is_cuda = prediction.is_cuda
        if not self.pre_train or target is None:
            loss = Variable(from_numpy(np.array([0.], dtype=np.float)))
        else:
            target = target if not is_cuda else target.cuda()
            loss = self.criterion.unstructured_loss(prediction, target)

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
            raise DeprecationWarning('Update for downscaling option!')
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
        loader = SimpleParallelLoader(dataset, num_workers=self.num_workers)
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

