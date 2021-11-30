import dataset_keys
import image_utils
from data_utils import image_hwc_to_chw, image_chw_to_hwc

MAX_TENSORBOARD_IMAGES = 6


def train_log(step, input_batch, output_batch, tensorboard_writer, **kwargs):
    if step % 100 == 0:
        loss = kwargs['loss'].detach().cpu().numpy()
        current_lr = kwargs['current_lr']
        print('step={}, loss: {:.12f}'.format(step, loss))
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Training/loss', loss, step)
            tensorboard_writer.add_scalar('Training/learning_rate', current_lr, step)

    if step % 2000 == 0 and tensorboard_writer is not None:
        rgb = input_batch[dataset_keys.COLOR_IMG].detach().cpu().numpy()
        gt_depth = input_batch[dataset_keys.HIGH_RES_DEPTH_IMG].detach().cpu().numpy().squeeze(1)
        depth_lowres = input_batch[dataset_keys.LOW_RES_DEPTH_IMG].detach().cpu().numpy().squeeze(1)
        valid_mask = input_batch[dataset_keys.VALID_MASK_IMG].detach().cpu().numpy().squeeze(1)
        pred_depth = output_batch[dataset_keys.PREDICTION_DEPTH_IMG].detach().cpu().numpy().squeeze(1)
        for i in range(min(rgb.shape[0], MAX_TENSORBOARD_IMAGES)):
            vmin = 0
            vmax = gt_depth.max()

            image_list = [image_chw_to_hwc(rgb[i]),
                          image_utils.colorize(gt_depth[i], vmin, vmax),
                          image_utils.colorize(depth_lowres[i], vmin, vmax),
                          image_utils.colorize(pred_depth[i], vmin, vmax),
                          image_utils.colorize(valid_mask[i], 0, 1)]

            h, w = pred_depth[i].shape[:2]
            montage = image_utils.create_montage_image(image_list, (w, h), grid_shape=(5, 1))
            tensorboard_writer.add_image(f'Training/{i}', image_hwc_to_chw(montage / 255), step)
