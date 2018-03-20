import os
import sys
import logging
import argparse
import yaml
import json

from skunkworks.trainers.learnedHC.visualization import VisualizationCallback

# FIXME needed to prevent segfault at import ?!
import vigra

NUM_WORKERS_PER_BATCH = 25
z_window_slice_training = None

from path import get_template_config_file


from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore

from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.multi_scale_loss import MultiScaleLossMaxPool
from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel, RemoveSegmentationFromTarget

from skunkworks.criteria.loss_transforms import GetMaskAndRemoveSegmentation
from skunkworks.criteria.multi_scale_loss_weighted import MultiScaleLossMaxPoolWeighted

import skunkworks.models as models

from skunkworks.postprocessing.watershed import DamWatershed

# Do we implement this in neurofire again ???
# from skunkworks.datasets.cremi.criteria import Euclidean, AsSegmentationCriterion

from skunkworks.datasets.cremi.loaders import get_cremi_loaders_realigned

# Import the different creiterions, we support.
# TODO generalized sorensen dice and tversky loss
from inferno.extensions.criteria import SorensenDiceLoss
from inferno.io.transform.base import Compose


# Structured stuff:
from skunkworks.trainers.learnedHC.trainHC import HierarchicalClusteringTrainer
from skunkworks.criteria.learned_HC import LHC
from skunkworks.metrics import LHCArandError

# from neurofire.criteria.multi_scale_loss

from skunkworks.postprocessing.pipelines import fixation_agglomerative_clustering_from_wsdt2d

# validation
from skunkworks.metrics import ArandErrorFromSegmentationPipeline

# multicut pipeline
from skunkworks.postprocessing.pipelines import local_affinity_multicut_from_wsdt2d

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory,
                    config,
                    data_config,
                    load_pretrained_model,
                    pretrain=False):
    VALIDATE_EVERY = (1, 'iterations') if pretrain else (70, 'iterations')
    SAVE_EVERY = (500, 'iterations') if pretrain else (300, 'iterations')
    # TODO: move these plots to tensorboard...?
    PLOT_EVERY = 10 if pretrain else 5 # This is only used by the struct. training

    # Get model
    if load_pretrained_model:
        model = Trainer().load(from_directory=project_directory,
                               filename='Weights/checkpoint.pytorch').model
    else:
        if pretrain:
            model_name = "PreTrainedUnet3DMultiscale"
            model_kwargs = config.get('pretrained_model_kwargs')
        else:
            model_name = "DynamicUNet3DMultiscale"
            model_kwargs = config.get('model_kwargs')
        model = getattr(models, model_name)(**model_kwargs)

    # Unstructed loss:
    affinity_offsets = data_config['volume_config']['segmentation']['affinity_offsets']

    # unstructured_loss = MultiScaleLossMaxPool(affinity_offsets,
    #                                           config['pretrained_model_kwargs']['scale_factor'],
    #                                           **config['loss_kwargs'])


    scaling_factor = config['pretrained_model_kwargs']['scale_factor']
    multiscale_loss_kwargs = config.get('multiscale_loss_kwargs', {})


    # Constantin approach:
    # loss = LossWrapper(criterion=SorensenDiceLoss(),
    #                    transforms=Compose(MaskTransitionToIgnoreLabel(affinity_offsets),
    #                                       RemoveSegmentationFromTarget()))
    #
    # unstructured_loss = MultiScaleLossMaxPool(loss,
    #                                         scaling_factor,
    #                                         invert_target=True,
    #                                         retain_segmentation=True,
    #                                         **multiscale_loss_kwargs)

    multiscale_loss = MultiScaleLossMaxPoolWeighted(SorensenDiceLoss(),
                                                    scaling_factor,
                                                    invert_target=True,
                                                    weighted_loss = False,
                                                    **multiscale_loss_kwargs)

    unstructured_loss = LossWrapper(criterion=multiscale_loss,
                        transforms=GetMaskAndRemoveSegmentation(affinity_offsets))

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.95


    # ----------
    # TRAINER:
    # ----------
    trainer = HierarchicalClusteringTrainer(model, pre_train=pretrain, **config)
    trainer.save_every(SAVE_EVERY, to_directory=os.path.join(project_directory, 'Weights'))
    trainer.build_criterion(LHC, options=config)
    trainer.register_unstructured_criterion(unstructured_loss)
    trainer.register_visualization_callback(VisualizationCallback(os.path.join(project_directory, 'Images'),plot_interval=PLOT_EVERY))
    trainer.build_optimizer(**config.get('training_optimizer_kwargs'))
    trainer.evaluate_metric_every('never')
    trainer.validate_every(VALIDATE_EVERY, for_num_iterations=1)
    trainer.register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))
    trainer.register_callback(AutoLR(factor=0.98,
                                  patience='100 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))

    # ----------
    # METRICS:
    # ----------
    # FIXME: MWS actually expects a prob. map, not affinities (....)
    # Use Mutex watershed for validation:
    # FIXME: now metric expects mutliscale...
    if pretrain:
        # Mutex watershed:
        # metric = ArandErrorFromSegmentationPipeline(DamWatershed(affinity_offsets, stride=[1, 2, 2],
        #                                                      n_threads=8))


        # Agglomeration from wsdt:
        init_segm_opts = config['HC_config']['init_segm']
        metric = ArandErrorFromSegmentationPipeline(
            fixation_agglomerative_clustering_from_wsdt2d(
                affinity_offsets,
                **init_segm_opts['wsdt_kwargs'],
                **init_segm_opts['prob_map_kwargs'])
        )

    else:
        # Use fixation clustering for validation: (incredibly ugly implementation atm:
        metric = LHCArandError(trainer.criterion)

    # Use multicut pipeline for validation (not working atm)
    # metric = ArandErrorFromSegmentationPipeline(local_affinity_multicut_from_wsdt2d(n_threads=10,
    #
    #                                                                                time_limit=120))

    trainer.build_metric(metric)

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=VALIDATE_EVERY).observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard, log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Trainer from checkpoint")
    trainer = Trainer().load(from_directory=os.path.join(project_directory, "Weights"))
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters=int(1e5),
             from_checkpoint=False,
             load_pretrained_model=False,
             pretrain=False):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_cremi_loaders_realigned(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_cremi_loaders_realigned(validation_configuration_file)

    # load network and training progress from checkpoint
    if from_checkpoint:
        logger.info("Loading trainer from checkpoint...")
        trainer = load_checkpoint(project_directory)
    else:
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config,
                                  load_pretrained_model,
                                  pretrain)

    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    logger.info("Lift off!")
    trainer.fit()


def make_train_config(train_config_file, offsets, gpus, nb_threads, reload_model=False):
    if not reload_model:
        template = get_template_config_file('./template_config_HC/train_config.yml', train_config_file)
        template['pretrained_model_kwargs']['out_channels'] = len(offsets)
        template['model_kwargs']['out_channels'] = len(offsets)
        template['HC_config']['offsets'] = offsets
        # TODO: modify number of inpuit dynamic channels...
    else:
        # Reload previous settings:
        template = yaml2dict(train_config_file)
    template['devices'] = gpus
    template['HC_config']['nb_threads'] = nb_threads
    # TODO: is this needed?
    template['HC_config']['batch_size'] = len(gpus)
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(data_config_file, offsets, n_batches, max_nb_workers, pretrain, reload_model=False):
    if not reload_model:
        template_path = './template_config_HC/data_config.yml' if not pretrain else './template_config_HC/pretrain/data_config.yml'
        template = get_template_config_file(template_path, data_config_file)
        template['volume_config']['segmentation']['affinity_offsets'] = offsets
    else:
        # Reload previous settings:
        template = yaml2dict(data_config_file)
    template['loader_config']['batch_size'] = n_batches
    num_workers = NUM_WORKERS_PER_BATCH * n_batches
    template['loader_config']['num_workers'] = num_workers if num_workers < max_nb_workers else max_nb_workers

    # Window size:
    default_wind_size = template['slicing_config']['window_size']['A']
    default_wind_size[0] = z_window_slice_training
    for dataset in template['slicing_config']['window_size']:
        template['slicing_config']['window_size'][dataset] = default_wind_size

    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def make_validation_config(validation_config_file, offsets, n_batches, max_nb_workers, pretrain, reload_model=False):
    if not reload_model:
        template_path = './template_config_HC/validation_config.yml' if not pretrain else './template_config_HC/pretrain/data_config.yml'
        template = get_template_config_file(template_path, validation_config_file)
        template['volume_config']['segmentation']['affinity_offsets'] = offsets
    else:
        # Reload previous settings:
        template = yaml2dict(validation_config_file)
    template['loader_config']['batch_size'] = n_batches
    num_workers = NUM_WORKERS_PER_BATCH * n_batches
    template['loader_config']['num_workers'] = num_workers if num_workers < max_nb_workers else max_nb_workers
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def parse_offsets(offset_file):
    assert os.path.exists(offset_file)
    with open(offset_file, 'r') as f:
        offsets = json.load(f)
    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('offset_file', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--max_nb_workers', type=int, default=int(8))
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))
    parser.add_argument('--pretrain', default='True')
    parser.add_argument('--nb_threads', default=int(8), type=int)
    parser.add_argument('--load_model', default='False')
    parser.add_argument('--z_window_size_training', default=int(15), type=int)
    parser.add_argument('--from_checkpoint', default='False')


    args = parser.parse_args()

    # Set the proper project folder:
    project_directory = args.project_directory
    pretrain = eval(args.pretrain)
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)
    if pretrain:
        project_directory = os.path.join(project_directory, "pre_train")
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    # We still leave options for varying the offsets
    # to be more flexible later variable
    offset_file = args.offset_file
    offsets = parse_offsets(offset_file)

    global z_window_slice_training
    z_window_slice_training = args.z_window_size_training

    max_nb_workers = args.max_nb_workers
    nb_threads = args.nb_threads


    # set the proper CUDA_VISIBLE_DEVICES env variables
    gpus = list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    load_model = eval(args.load_model)


    train_config = os.path.join(project_directory, 'train_config.yml')
    make_train_config(train_config, offsets, gpus, nb_threads, reload_model=load_model)

    data_config = os.path.join(project_directory, 'data_config.yml')
    make_data_config(data_config, offsets, len(gpus), max_nb_workers, pretrain, reload_model=load_model)

    validation_config = os.path.join(project_directory, 'validation_config.yml')
    make_validation_config(validation_config, offsets, len(gpus), max_nb_workers, pretrain, reload_model=load_model)

    print("Pretrain: {}; Load model: {}".format(pretrain,load_model))

    training(project_directory,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters,
             from_checkpoint=eval(args.from_checkpoint),
             pretrain=pretrain,
             load_pretrained_model=load_model)


if __name__ == '__main__':
    main()
