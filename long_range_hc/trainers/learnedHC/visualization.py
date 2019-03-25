import matplotlib
matplotlib.use('Agg')
import os
# import seaborn
from matplotlib import pyplot as plt
import vigra
from os.path import join
import numpy as np
import atexit
import copy
import dill
from torch import save as torch_save
#from vigra.analysis import regionImageToCrackEdgeImage
from inferno.utils.io_utils import toh5

from multiprocessing import Queue, Process
from inferno.trainers.callbacks.base import Callback

matplotlib.rcParams.update({'font.size': 5})
# np.random.seed(1236)
# fixed_rand = np.random.rand(10000, 3)
# fixed_rand[0, :] = 0
# fixed_rand[1, :] = [1., 0., 0.]
# fixed_rand[2, :] = [0., 1., 0.]
# fixed_rand[3, :] = [0., 0., 1.]
# rand_cm = matplotlib.colors.ListedColormap(fixed_rand, name='indexed')
rand_cm = matplotlib.colors.ListedColormap(np.random.rand(1000000, 3))

DEF_INTERP = 'none'
segm_plot_kwargs = {'vmax': 1000000, 'vmin':0}

from segmfriends.transform.segm_to_bound import compute_boundary_mask_from_label_image
from segmfriends.transform.inferno.temp_crap import FindBestAgglFromOversegmAndGT
from segmfriends.features import from_affinities_to_hmap


class VisualizationCallback(Callback):
    def __init__(self, image_folder, plot_interval=1, batches=None):
        super(VisualizationCallback, self).__init__()
        if batches is None:
            self.batches = [0]
        else:
            self.batches = batches
        self.vis = Visualizer()
        self.val_counter = 0
        self.plot_interval = plot_interval
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        self.image_folder = image_folder

    def get_observe_variables(self):
        return ['label_image']

    def check_plot(self):
        return self.trainer._iteration_count % self.plot_interval == 0

    def to_numpy(self, data, batch):
        return dict((k, v[batch].data.cpu().numpy()) for k, v in data.items() if v is not None)

    def plot_batch_pretraining(self, data, batch, suffix='', file_ext='pdf'):
        """
        :param data: dictionary of Variables (first expected dimension is batch_channel)
        :param batch: number of the batch
        :param suffix: suffix string for the plot_name
        :param file_ext:
        """
        img_data = self.to_numpy(data, batch)

        self.vis.draw('pretrain_plots', img_data,
                      self.image_folder,
                      "preTrain_iter_{:0>8}_{}_{}.{}".format(self.trainer._iteration_count, batch,
                                                                suffix, file_ext)
                      )


    def enqueue_batch_plots(self, image_data, batch_list=None, suffix='', file_ext='pdf'):
        """
        :param image_data: list of dictionaries containing numpy arrays (one for each batch)
        :param batch_list: if not None, is a list of plotted batches
        :param suffix: suffix string for the plot_name
        :param file_ext:
        """
        if batch_list is None:
            batch_list = range(len(image_data))
        else:
            assert isinstance(batch_list, list)

        for batch in batch_list:
            self.vis.draw('segm_results', image_data[batch],
                          self.image_folder,
                          "segmResults_iter_{:0>8}_{}_{}.{}".format(self.trainer._iteration_count, batch,
                                                                      suffix, file_ext)
                          )
            self.vis.draw('dyn_predictions', image_data[batch],
                          self.image_folder,
                          "dynPredics_iter_{:0>8}_{}_{}.{}".format(self.trainer._iteration_count, batch,
                                                                    suffix, file_ext)
                          )


    def end_of_training_iteration(self, **_):
        return

        # if not self.check_plot():
        #     return
        #
        # # TODO: update images on tensorboard
        # self.trainer.save_model()
        # output_file = join(self.image_folder, "../Weights/static_model.pytorch")
        # torch_save(self.trainer.model.static_net, output_file, pickle_module=dill)
        #
        # # for batch in self.batches:
        # #     inputs = self.trainer.get_state('training_inputs')[batch].cpu().numpy()
        # #     prediction = self.trainer.get_state('training_prediction')[batch].cpu().numpy()
        # #     self.plot_batch(inputs, prediction, batch, suffix="fin")
        #
        # # self.trainer.update_state('label_image', self.trainer.criterion[batch].get_display_label_image())

    def get_config(self):
        config = super(VisualizationCallback, self).get_config()
        config.update({'vis': None})
        return config

    def set_config(self, config_dict):
        config_dict['vis'] = Visualizer()
        return super(VisualizationCallback, self).set_config(config_dict)

class Visualizer(object):
    def __init__(self):
        self.draw_queue = Queue(maxsize=100)
        self.draw_daemon = Process(target=save_images_multi, args=((self.draw_queue),))
        self.draw_daemon.start()

        def cleanup():
            # clear queue
            while not self.draw_queue.empty():
                self.draw_queue.get()
            print("sending kill message to Vizualizer")
            self.close()
        atexit.register(cleanup)

    def draw(self, plot_type, img_list, path, name, column_size=0):
        self.draw_queue.put([plot_type, copy.deepcopy(img_list), path, name, column_size])

    def close(self):
        self.draw_queue.put(("kill", "", "", "", 0))
        self.draw_daemon.join()
        self.draw_daemon.terminate()
        
    def get_config(self):
        config = super(Visualizer, self).get_config()
        config.update({'draw_queue': None})
        config.update({'draw_daemon': None})
        return config

    def set_config(self, state_dict):
        draw_q = Queue(maxsize=100)
        draw_d = Process(target=save_images_multi, args=((self.draw_queue),))
        self.d.start()
        state_dict.update({'draw_queue': draw_q})
        state_dict.update({'draw_queue': draw_d})
        super(Visualizer, self).set_config(state_dict)


def save_images_multi(queue):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    matplotlib.rcParams.update({'font.size': 5})
    np.random.seed(1236)
    fixed_rand = np.random.rand(10000, 3)
    fixed_rand[0, :] = 0
    fixed_rand[1, :] = [1., 0., 0.]
    fixed_rand[2, :] = [0., 1., 0.]
    fixed_rand[3, :] = [0., 0., 1.]
    # TODO: why here...?
    rand_cm = matplotlib.colors.ListedColormap(fixed_rand, name='indexed')

    not_dead = True

    while not_dead:
        plot_type, img_data, path, name, column_size = queue.get()
        if plot_type == "kill":
            not_dead = False
            break
        # elif plot_type == "segm_results":
        #     plot_segm_results(img_data, path, name)
        else:
            plot_data_batch(img_data, plot_type, path, name)





def plot_segm_results(img_data, targets, z_slice):
    targets[0, z_slice].matshow(img_data['raw'][z_slice], cmap='gray', interpolation=DEF_INTERP)
    # targets[1, z_slice].matshow(img_data['predictions_merge_prio'][0][9][z_slice], cmap='gray',
    #                             interpolation=DEF_INTERP)
    # targets[2, z_slice].matshow(1. - img_data['predictions_notMerge_prio'][0][9][z_slice], cmap='gray',
    #                             interpolation=DEF_INTERP)

    targets[1, z_slice].matshow(img_data['prob_map'][z_slice], cmap='gray', interpolation=DEF_INTERP)
    targets[2, z_slice].matshow(img_data['new_prob_map'][z_slice], cmap='gray', interpolation=DEF_INTERP)



    # bound_init_segm = plot_segm(targets[1, z_slice], img_data['init_segm'], z_slice=z_slice, background=img_data['raw'])

    GT = img_data['GT_labels']
    if GT is not None:


        target1 = targets[3, z_slice]

        # masked_ignore_label = GT == 0
        # plot_segm(target1, mask_array(GT, masked_ignore_label), z_slice, background=img_data['raw'])
        # target1.matshow(get_masked_boundary_mask(GT)[z_slice], cmap='Set1', alpha=0.9, interpolation=DEF_INTERP)

        masked_ignore_label = img_data['best_aggl'] == 0
        plot_segm(target1, mask_array(img_data['best_aggl'], masked_ignore_label), z_slice,
                  background=img_data['raw'])
        target1.matshow(get_masked_boundary_mask(img_data['init_segm'])[z_slice], cmap='gray', alpha=0.7,
                        interpolation=DEF_INTERP)
        target1.matshow(get_masked_boundary_mask(GT)[z_slice], cmap='Set1', alpha=0.9, interpolation=DEF_INTERP)
        # target1.set_title("GT labels (red bound.) and initial oversegm (black)")


        # plot_segm(targets[3, z_slice], mask_array(img_data['pred_segm'], masked_ignore_label), z_slice,
        #           background=img_data['predictions_merge_prio'][0][9])
        plot_segm(targets[4, z_slice], img_data['pred_segm'], z_slice,
                  background=img_data['raw'])


        # target2 = targets[2, z_slice]
        # find_splits_merges(target2, mask_array(img_data['best_aggl'], masked_ignore_label),
        #                    mask_array(img_data['pred_segm'], masked_ignore_label), z_slice,
        #                    background=img_data['prob_map'])

        # targets[5, z_slice].matshow(img_data['predictions_merge_prio'][0][9][z_slice], cmap='gray', interpolation=DEF_INTERP)

        if 'loss_weights' in img_data:
            targets[5, z_slice].matshow(1. - img_data['predictions_notMerge_prio'][0][9][z_slice], cmap='gray',
                                    interpolation=DEF_INTERP)
        else:
            plot_segm(targets[5, z_slice], img_data['final_segm_pretrained'], z_slice,
                      background=img_data['raw'])



        # plot_segm(targets[3,z_slice], mask_array(img_data['pred_segm'], masked_ignore_label), z_slice, background=img_data['predictions_merge_prio'][0][6], highlight_boundaries=False,
        #           plot_label_colors=False)



    # FIXME: this prints the UCM at milestep 0, not the final one...!
    # targets[4, z_slice].matshow(mask_the_mask(img_data['final_UCM'][1][z_slice], interval=[-2., 0.001]), cmap='jet', interpolation=DEF_INTERP, vmin=0, vmax=1)

    if 'loss_weights' in img_data:
        targets[6, z_slice].matshow(mask_the_mask(img_data['loss_weights'][0][9,z_slice], value_to_mask=0.), cmap='jet',
                                interpolation=DEF_INTERP, vmin=0, vmax=5)
    else:
        find_splits_merges(targets[6, z_slice], img_data['best_aggl'],
                           img_data['pred_segm'], z_slice,
                           background=img_data['prob_map'])

def plot_3_classes(target, segm, z_slice=0, background=None, mask_value=None, highlight_boundaries=True):
    """Shape of expected background: (z,x,y)"""
    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)

    if mask_value is not None:
        segm = mask_the_mask(segm, value_to_mask=mask_value)
    target.matshow(mask_the_mask(segm[z_slice]==1., value_to_mask=0.), cmap='summer', interpolation=DEF_INTERP, alpha=0.4, **segm_plot_kwargs)
    target.matshow(mask_the_mask(segm[z_slice]==2., value_to_mask=0.), cmap='autumn', interpolation=DEF_INTERP, alpha=0.4, **segm_plot_kwargs)
    target.matshow(mask_the_mask(segm[z_slice]==3., value_to_mask=0.), cmap='winter', interpolation=DEF_INTERP, alpha=0.4, **segm_plot_kwargs)
    masked_bound = get_masked_boundary_mask(segm)
    # if highlight_boundaries:
    #     target.matshow(masked_bound[z_slice], cmap='gray', alpha=0.6, interpolation=DEF_INTERP)
    return masked_bound

def plot_dyn_predictions(img_data, targets, z_slice, milestep):


    plot_offset = 0 if z_slice==1 else 3
    for i, offset in enumerate([0,7,8,9,15,16]):
        plot_3_classes(targets[i, plot_offset], img_data['prediction_labels'][offset]+1, z_slice, background=img_data['raw'])
        if 'merge_targets' in img_data:
            plot_3_classes(targets[i, plot_offset+1], img_data['GT_classes_targets'][offset], z_slice, background=img_data['raw'],
                  mask_value=0.)

        targets[i, plot_offset + 2].matshow(img_data['raw'][z_slice], cmap='gray', interpolation=DEF_INTERP)
        targets[i, plot_offset+2].matshow(mask_the_mask(img_data['predictions_merge_prio'][0][offset, z_slice]<0.5, value_to_mask=0.), cmap='autumn',
                                              interpolation=DEF_INTERP)


def plot_pretrain_predictions(img_data, targets, z_slice, fig=None):
    if 'rnd_ints' in img_data and z_slice == 0:
        print(img_data['rnd_ints'])
        if img_data['rnd_ints'][0] == 0:
            str = 'Only RAW'
        elif img_data['rnd_ints'][0] == 1:
            str = 'RAW + initSegm'
        elif img_data['rnd_ints'][0] == 2:
            str = 'RAW + initSegm + lookAheads'
        else:
            raise NotImplementedError
        fig.suptitle(str)

    if 'final_segm' in img_data:
        # Plot 1:
        target1 = targets[0, z_slice]
        finalSegm = img_data['final_segm']
        gt = img_data['target'][0]
        plot_segm(target1, finalSegm, z_slice,
                  background=img_data['raw'])
        target1.matshow(get_masked_boundary_mask(img_data['init_segm'])[z_slice], cmap='gray', alpha=0.7,
                        interpolation=DEF_INTERP)
        target1.matshow(get_masked_boundary_mask(gt)[z_slice], cmap='Set1', alpha=0.9, interpolation=DEF_INTERP)

        plot_segm(targets[1, z_slice], gt, z_slice, mask_value=0,
                  background=img_data['raw'], highlight_boundaries=True)


        # plot_segm(targets[0, z_slice], img_data['final_segm'], z_slice, mask_value=0,
        #           background=img_data['raw'])
        # plot_segm(targets[1, z_slice], img_data['target'][0], z_slice, mask_value=0,
        #           background=img_data['raw'])

        # Plot 2:
        # find_best = FindBestAgglFromOversegmAndGT(border_thickness=2,
        #                                           number_of_threads=8,
        #                                           break_oversegm_on_GT_borders=True,
        #                                           undersegm_threshold=6000)
        #
        # undersegm_mask = np.logical_and(find_best(finalSegm, gt) == 0, gt != 0)
        # oversegm_mask = np.logical_and(find_best(gt, finalSegm) == 0, gt != 0)
        # masks = undersegm_mask + oversegm_mask*2
        # plot_3_classes(targets[1, z_slice], masks, z_slice=z_slice, background=img_data['raw'],mask_value=0.)
        # targets[1, z_slice].matshow(get_masked_boundary_mask(finalSegm)[z_slice], cmap='Set1', alpha=0.9, interpolation=DEF_INTERP)
    else:
        if 'init_segm' in img_data:
            if 'lookAhead2' in img_data:
                plot_segm(targets[0, z_slice], img_data['init_segm'], z_slice, mask_value=0,
                          background=img_data['raw'])
                targets[0, z_slice].matshow(get_masked_boundary_mask(img_data['lookAhead1'])[z_slice], cmap='summer', alpha=0.9,
                                            interpolation=DEF_INTERP)
                targets[0, z_slice].matshow(get_masked_boundary_mask(img_data['lookAhead2'])[z_slice], cmap='Set1', alpha=1., interpolation=DEF_INTERP)

                plot_segm(targets[1, z_slice], img_data['target'][0], z_slice, mask_value=0,
                          background=img_data['raw'])
            elif 'lookAhead1' in img_data:
                plot_segm(targets[0, z_slice], img_data['lookAhead1'], z_slice, mask_value=0,
                          background=img_data['raw'], highlight_boundaries=False)
                targets[0, z_slice].matshow(get_masked_boundary_mask(img_data['init_segm'])[z_slice], cmap='gray',
                                            alpha=1.,
                                            interpolation=DEF_INTERP)
                targets[0, z_slice].matshow(get_masked_boundary_mask(img_data['lookAhead1'])[z_slice], cmap='Set1',
                                            alpha=1., interpolation=DEF_INTERP)

                plot_segm(targets[1, z_slice], img_data['target'][0], z_slice, mask_value=1,
                          background=img_data['raw'], highlight_boundaries=True)
                targets[1, z_slice].matshow(get_masked_boundary_mask(img_data['lookAhead1'])[z_slice], cmap='Set1',
                                            alpha=1., interpolation=DEF_INTERP)
                # targets[1, z_slice].matshow(get_masked_boundary_mask(img_data['target'][0])[z_slice], cmap='summer',
                #                             alpha=1., interpolation=DEF_INTERP)
            else:
                plot_segm(targets[0, z_slice], img_data['init_segm'], z_slice, mask_value=0,
                          background=img_data['raw'], highlight_boundaries=True)
                # targets[0, z_slice].matshow(get_masked_boundary_mask(img_data['target'][0])[z_slice], cmap='Set1',
                #                             alpha=0.6, interpolation=DEF_INTERP)

                plot_segm(targets[1, z_slice], img_data['target'][0], z_slice, mask_value=0,
                          background=img_data['raw'], highlight_boundaries=True)
        else:
            plot_segm(targets[0, z_slice], img_data['target'][0], z_slice, mask_value=0,
                      background=img_data['raw'], highlight_boundaries=True)
            targets[1, z_slice].matshow(img_data['stat_prediction'][0, z_slice], cmap='gray',
                                            interpolation=DEF_INTERP)
            if 'loss_weights' in img_data:
                cax = targets[1, z_slice].matshow(mask_the_mask(img_data['loss_weights'][0, z_slice], interval=[-0.0001, 1.0001]),
                                            cmap='rainbow',
                                            alpha=0.7, interpolation=DEF_INTERP)
            targets[1, z_slice].matshow(mask_the_mask(img_data['target'][0 + 1, z_slice], value_to_mask=1.0),
                                            cmap='Set1',
                                            alpha=0.3, interpolation=DEF_INTERP)



    nb_offs = img_data['stat_prediction'].shape[0]
    if  nb_offs == 12:
        plotted_offsets = [4,5,8,11,0]
    elif nb_offs == 27:
        plotted_offsets = [3, 5, 14, 25, 0]
    elif nb_offs == 17:
        plotted_offsets = [1, 0, 7, 8, 16]
    elif nb_offs == 3:
        plotted_offsets = [0, 1, 2]
    else:
        raise NotImplementedError()

    for i, offset in enumerate(plotted_offsets):
        targets[i+2, z_slice].matshow(img_data['stat_prediction'][offset,z_slice], cmap='gray', interpolation=DEF_INTERP)
        # targets[i+1, z_slice].matshow(img_data['target'][offset+1, z_slice], cmap='gray', interpolation=DEF_INTERP)
        if 'loss_weights' in img_data:
            cax = targets[i + 2, z_slice].matshow(mask_the_mask(img_data['loss_weights'][offset, z_slice], value_to_mask=1.),
                                            cmap='rainbow',
                                            alpha=0.3, interpolation=DEF_INTERP)
            if z_slice == 6 and i == 0:
                fig.colorbar(cax, ax=targets[i + 2, z_slice])
        targets[i+2, z_slice].matshow(mask_the_mask(img_data['target'][offset+1, z_slice], value_to_mask=1.0), cmap='Set1',
                                      alpha=0.3, interpolation=DEF_INTERP)

    # Validation plot:
    if 'final_segm' in img_data and 'lookAhead1' in img_data and 'lookAhead2' in img_data:
        targets[2, z_slice].matshow(get_masked_boundary_mask(img_data['lookAhead1'])[z_slice], cmap='summer', alpha=1.,
                                    interpolation=DEF_INTERP)
        targets[3, z_slice].matshow(get_masked_boundary_mask(img_data['lookAhead2'])[z_slice], cmap='summer', alpha=1.,
                                    interpolation=DEF_INTERP)






def plot_data_batch(img_data, plot_type, path, name, column_size=0):
    # # DUMP DATA FOR PLOT-DEBUGGING:
    # import pickle
    # a = [img_data, path, name, column_size]
    #
    # file_Name = "dumped_plot_data_postprocessing3"
    # fileObject = open(file_Name, 'wb')
    #
    # # this writes the object a to the
    # # file named 'testfile'
    # pickle.dump(a, fileObject)
    #
    # # here we close the fileObject
    # fileObject.close()
    # print("Dumped!")
    # # else:
    raw = img_data['raw']
    z_context = raw.shape[0] if raw.shape[0] <= 7 else 7

    ncols = 6 if plot_type == "dyn_predictions" else 7
    nrows = 6 if plot_type == "dyn_predictions" else 7


    # It could be inverted, but anyway now they are the same:
    f, ax = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize=(ncols, nrows))
    for a in f.get_axes():
        a.axis('off')

    nb_milesteps = img_data['edge_indicators'].shape[0] -1 if plot_type == "dyn_predictions" else None

    if plot_type=="dyn_predictions":
        assert nb_milesteps==1
        img_data['all_predictions'] = np.concatenate([img_data['predictions_merge_prio'][[1]],
                                                img_data['predictions_notMerge_prio'],
                                                img_data['predictions_inner']])
        img_data['prediction_labels'] = np.argmax(img_data['all_predictions'], axis=0)

        # Targets:
        if 'merge_targets' in img_data:
            img_data['GT_classes_targets'] = (img_data['merge_targets'] + 2. * img_data['split_targets'] + 3. * img_data['inner_targets'])[0]
    elif plot_type=="segm_results":
        # Compute prob. map prediction:
        img_data['new_prob_map'] = from_affinities_to_hmap(1.-img_data['predictions_notMerge_prio'][0],
                                                    img_data['offsets'],
                                                    img_data['prob_map_kwargs'].get('used_offsets', [1, 2]),
                                                    img_data['prob_map_kwargs'].get('offset_weights', [1., 1.]))
        if 'merge_targets' not in img_data:
            f.suptitle("Structrued prediction: CS {}; VIs {}; VIm {}; ARAND {}\n".format(*img_data['scores']) + "Init. segm. + pretrained: {}".format(img_data['scores_pretrained']), fontsize=5, )

    img_data['target'][0], _, _ = vigra.analysis.relabelConsecutive(img_data['target'][0].astype('uint32'))
    for z_slice in range(z_context):
        if plot_type == "segm_results":
            plot_segm_results(img_data,ax,z_slice)
        elif plot_type == "dyn_predictions":
            # for milestep in range(1):
            #     plot_dyn_predictions(img_data,ax,z_slice,milestep)
            # print(nb_milesteps)
            if z_slice==1 or z_slice==5:
                plot_dyn_predictions(img_data,ax,z_slice,0)
        elif plot_type == "pretrain_plots":
            plot_pretrain_predictions(img_data,ax,z_slice, fig=f)
        else:
            raise NotImplementedError()

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    if name.endswith('pdf'):
        f.savefig(join(path, name), format='pdf')
    else:
        f.savefig(join(path, name), dpi=400)
    # plt.show()

    plt.clf()
    plt.close('all')


"""
HELPER FUNCTIONS
"""


def mask_the_mask(mask, value_to_mask=0., interval=None):
    if interval is not None:
        return np.ma.masked_where(np.logical_and(mask < interval[1], mask > interval[0]), mask)
    else:
        return np.ma.masked_where(np.logical_and(mask < value_to_mask+1e-3, mask > value_to_mask-1e-3), mask)

def mask_array(array_to_mask, mask):
    return np.ma.masked_where(mask, array_to_mask)

def get_bound_mask(segm):
    # print("B mask is expensive...")
    return compute_boundary_mask_from_label_image(segm,
                                                  np.array([[0,1,0], [0,0,1]]),
                                                  compress_channels=True)

def get_masked_boundary_mask(segm):
    #     bound = np.logical_or(get_boundary_mask(segm)[0, 0],get_boundary_mask(segm)[1, 0])
    bound = get_bound_mask(segm)
    return mask_the_mask(bound)

def plot_segm(target, segm, z_slice=0, background=None, mask_value=None, highlight_boundaries=True, plot_label_colors=True):
    """Shape of expected background: (z,x,y)"""
    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)

    if mask_value is not None:
        segm = mask_the_mask(segm,value_to_mask=mask_value)
    if plot_label_colors:
        target.matshow(segm[z_slice], cmap=rand_cm, alpha=0.4, interpolation=DEF_INTERP, **segm_plot_kwargs)
    masked_bound = get_masked_boundary_mask(segm)
    if highlight_boundaries:
        target.matshow(masked_bound[z_slice], cmap='gray', alpha=0.6, interpolation=DEF_INTERP)
    return masked_bound

def find_splits_merges(target, GT_labels, segm, z_slice=0, background=None):
    GT_bound = get_bound_mask(GT_labels) * 3.
    segm_bound = get_bound_mask(segm) * (1.)
    diff_bound = (GT_bound+segm_bound).astype(np.int32)

    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)
    target.matshow(mask_the_mask(diff_bound==4)[z_slice], cmap='summer', alpha=1, interpolation=DEF_INTERP)
    target.matshow(mask_the_mask(diff_bound == 1)[z_slice], cmap='winter', alpha=1,
                interpolation=DEF_INTERP)
    target.matshow(mask_the_mask(diff_bound == 3)[z_slice], cmap='autumn', alpha=1,
                interpolation=DEF_INTERP)





def plot_output_affin(target, out_affin, nb_offset=1, z_slice=0):
    # Select the ones along x:
    cax = target.matshow(out_affin[nb_offset,z_slice,:,:], cmap=plt.get_cmap('seismic'), vmin=0, vmax=1, interpolation=DEF_INTERP)

def plot_affs_divergent_colors(ax, out_affin, type='pos', z_slice=0):
    # Select the ones along x:
    if type=='pos':
        data = out_affin * (out_affin>0. )
    else:
        data = out_affin * (out_affin < 0.)

    cax = ax.matshow(mask_the_mask(data[0,z_slice,:,:]), cmap=plt.get_cmap('autumn'), interpolation=DEF_INTERP)
    # cax = ax.matshow(mask_the_mask(neg_out_affin[axis, z_slice, :, :]), cmap=plt.get_cmap('cool'),
    #                  interpolation=DEF_INTERP)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def plot_lookahead(ax, lookahead, mergers=True, z_slice=0):
    # cax = ax.matshow(mask_the_mask(lookahead[z_slice, :, :, 1], value_to_mask=-2.0), cmap=plt.get_cmap('viridis'),
    #                  interpolation=DEF_INTERP)
    channel = 0 if mergers else 1
    cax = ax.matshow(mask_the_mask(lookahead[z_slice, :, :, channel], value_to_mask=-2.0), cmap=plt.get_cmap('jet'),
                     interpolation=DEF_INTERP)
    if mergers:
        mask_alive_boundaries = lookahead[z_slice, :, :, 1] > - 2.0
        cax = ax.matshow(mask_the_mask(mask_alive_boundaries),
                         cmap=plt.get_cmap('spring'),
                         interpolation=DEF_INTERP)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def get_figure(ncols, nrows, hide_axes=True):
    f, ax = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize=(ncols, nrows))
    for a in f.get_axes():
        a.axis('off')
    return f, ax

def save_plot(f, path, file_name):
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    if file_name.endswith('pdf'):
        f.savefig(os.path.join(path, file_name), format='pdf')
