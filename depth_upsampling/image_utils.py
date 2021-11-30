import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_montage_image(image_list, image_shape=(640, 480), grid_shape=(3, 1)) -> np.ndarray:
    height, width = image_shape[1], image_shape[0]
    montage = np.zeros((image_shape[1] * grid_shape[1], image_shape[0] * grid_shape[0], 3), dtype="uint8")

    x_shift = 0
    y_shift = 0

    for n in range(len(image_list)):
        image = cv2.resize(image_list[n], (width, height), interpolation=cv2.INTER_NEAREST)
        montage[y_shift * height: (y_shift + 1) * height, x_shift * width: (x_shift + 1) * width] = image
        x_shift += 1

        if x_shift % (grid_shape[0]) == 0 and x_shift > 0:
            y_shift += 1
            x_shift = 0

    return montage


def colorize(image, vmin=None, vmax=None, cmap='turbo'):

    vmin = image.min() if vmin is None else vmin
    vmax = image.max() if vmax is None else vmax

    if vmin != vmax:
        image = (image - vmin) / (vmax - vmin)
    else:
        image = image * 0.

    cmapper = plt.cm.get_cmap(cmap)
    image = cmapper(image, bytes=True)

    img = image[:, :, :3]

    return img
