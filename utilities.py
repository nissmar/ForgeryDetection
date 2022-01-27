from PIL import Image
import numpy as np
import skimage.measure


def show_heat_map(im, mask):
    """combines bw image ([0,255]) with mask ([0,1])"""
    mask = Image.fromarray((255 * mask).astype("uint8"))
    mask = np.array(mask.resize((im.shape[1], im.shape[0]))) / 255

    res = np.repeat(im[:, :, np.newaxis], 3, axis=2) / 255
    res[:, :, 1] = res[:, :, 0] * (1 - mask)
    res[:, :, 2] = res[:, :, 0] * (1 - mask)
    return res


def reduce_array(arr, filter, method=np.max):
    return skimage.measure.block_reduce(arr, (filter, filter), method)
