import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import ARKitScenesDataset
from data_utils import image_chw_to_hwc
import dataset_keys


def sample_vis(dataset_path: str, split: str, sample_id: str, max_depth):
    dataset = ARKitScenesDataset(root=dataset_path, split=split)
    video_id = sample_id.split('_')[0]
    idx = None
    for i in range(len(dataset)):
        if dataset.samples[i][0] == video_id and dataset.samples[i][1] == sample_id:
            idx = i
            break
    if idx is None:
        raise ValueError(f'Can\'t find sample from split={split}, with video_id={video_id} and sample_id={sample_id}')
    sample = dataset[idx]

    max_depth = np.min([max_depth,
                        sample[dataset_keys.HIGH_RES_DEPTH_IMG].max(),
                        sample[dataset_keys.LOW_RES_DEPTH_IMG].max()])
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_title('Color img')
    axes[0, 0].axis(False)
    axes[0, 0].imshow(image_chw_to_hwc(sample[dataset_keys.COLOR_IMG]/255))
    axes[0, 1].set_title('High Res img (0=no depth)')
    axes[0, 1].axis(False)
    img = axes[0, 1].imshow(sample[dataset_keys.HIGH_RES_DEPTH_IMG][0], vmin=0, vmax=max_depth, cmap=plt.get_cmap('turbo'))
    fig.colorbar(img, ax=axes[0, 1])
    axes[1, 1].set_title('Low Res img')
    axes[1, 1].axis(False)
    img = axes[1, 1].imshow(sample[dataset_keys.LOW_RES_DEPTH_IMG][0], vmin=0, vmax=max_depth, cmap=plt.get_cmap('turbo'))
    fig.colorbar(img, ax=axes[1, 1])
    axes[1, 0].set_title('Color and low res overlay')
    axes[1, 0].axis(False)
    axes[1, 0].imshow(image_chw_to_hwc(sample[dataset_keys.COLOR_IMG]/255))
    axes[1, 0].imshow(sample[dataset_keys.LOW_RES_DEPTH_IMG][0], vmin=0, vmax=max_depth, cmap=plt.get_cmap('turbo'), alpha=0.5)
    plt.show()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="path to the dataset")
    parser.add_argument("split", choices=['train', 'val'], type=str, help="sample split (train/val)")
    parser.add_argument("sample_id", type=str, help="the id of the sample")
    parser.add_argument("--max_depth", type=float, default=5, help="clip the depth image to max depth [meters]")

    args = parser.parse_args()

    sample_vis(args.data_path, args.split, args.sample_id, args.max_depth)
