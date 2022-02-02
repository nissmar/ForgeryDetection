from PIL import Image
import numpy as np
import skimage.measure
from torch import FloatTensor


def show_heat_map(im, mask, mode=None):
    """combines bw image ([0,255]) with mask ([0,1])"""

    mask = Image.fromarray((255 * mask).astype("uint8"))
    mask = np.array(mask.resize((im.shape[1], im.shape[0]))) / 255
    res = np.repeat(im[:, :, np.newaxis], 3, axis=2) / 255

    if mode == "MASK":
        res_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        res_mask[:, :, 1:] = 0
        alpha = 0.1
        return res * res_mask * alpha + (res + res_mask) / 2 * (1 - alpha)

    res[:, :, 1] = res[:, :, 0] * (1 - mask)
    res[:, :, 2] = res[:, :, 0] * (1 - mask)
    return res


def reduce_array(arr, filter, method=np.max):
    return skimage.measure.block_reduce(arr, (filter, filter), method)


def load_tensor_img(str):
    im = np.array(Image.open(str))
    im = np.array(im).mean(axis=2)  # convert to bw
    im = FloatTensor(im)
    return im


def Fmeasure(targ, estim):
    estim = Image.fromarray((255 * estim).astype("uint8"))
    estim = np.array(estim.resize((targ.shape[1], targ.shape[0]))) / 255
    estim = estim > 0.5
    targ = targ > 255 / 2
    tp = np.sum((targ == 1) * (estim == 1))
    fp = np.sum((targ == 0) * (estim == 1))
    fn = np.sum((targ == 1) * (estim == 0))
    return (2 * tp) / (2 * tp + fn + fp)


def curve_score(Targx, Targy, Testx, Testy):
    """custom score that measures how low is Test compared to Target"""
    sc = 1
    for x, y in zip(Testx, Testy):
        if x < Targx[0]:  # x>Testx
            sc += 0
        elif x >= Targx[-1]:
            sc += 0
        else:
            ind = [e > x for e in Targx].index(1)  # first item > x in Targx

            a1 = Targx[ind] - x
            a2 = x - Targx[ind - 1]

            targ_y = (Targy[ind] * a2 + Targy[ind - 1] * a1) / (a2 + a1)

            sc = min(y / targ_y, sc)
    return 1 - sc


def compute_suspicious_pixels(M, im, patch_size=64, e_per_bin=3000, f_var=0.1):
    step = patch_size // 2
    n, m = im.shape
    new_n = len(range(0, n - patch_size, step))
    new_m = len(range(0, m - patch_size, step))
    percent_wrong = np.zeros((new_n, new_m))
    Lm, Lv = M.image_variance_hist(im, e_per_bin=100000, f_var=0.005)
    for i in range(0, n - patch_size, step):
        for j in range(0, m - patch_size, step):
            Lm2, Lv2 = M.image_variance_hist(
                im[i : i + patch_size, j : j + patch_size],
                e_per_bin=e_per_bin,
                f_var=f_var,
            )
            percent_wrong[i // step, j // step] = curve_score(Lm, Lv, Lm2, Lv2)
    return percent_wrong
