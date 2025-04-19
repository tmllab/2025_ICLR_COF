import numpy as np
import torch
from PIL import Image
from torchvision import transforms

encode_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)


def convert_decode_to_pil(rec_image):
    rec_image = 2.0 * rec_image - 1.0
    rec_image = torch.clamp(rec_image, -1.0, 1.0)
    rec_image = (rec_image + 1.0) / 2.0
    rec_image *= 255.0
    rec_image = rec_image.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in rec_image]
    return pil_images


def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 * C)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    in_chans = imgs.shape[1]
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
    return x


def unpatchify(x, p):
    """
    x: (N, L, patch_size**2 * C)
    imgs: (N, C, H, W)
    """
    # p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs


def find_most_relevant(codebooks, test_indices):
    return


def cal_metric(pred_arr, gt_arr):
    intersection = np.logical_and(pred_arr, gt_arr).sum()
    union = np.logical_or(pred_arr, gt_arr).sum()
    iou = intersection / union

    # Cal foreground acc
    foreground_acc = intersection / sum(gt_arr.flatten() == 1)

    # Cal image pixel-level acc, (correct_fore + correct_back) / all_pixel
    pixel_acc = (sum(np.logical_and((pred_arr.flatten()==1), (gt_arr.flatten() == 1))) + sum(np.logical_and((pred_arr.flatten()==0), (gt_arr.flatten() == 0)))) / (256 * 256)
    black_rate = sum(pred_arr.flatten() == 0) / (256 * 256)
    return iou, foreground_acc, pixel_acc, black_rate


def jaccard_similarity(l1, l2):
    set1 = set(l1)
    set2 = set(l2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def codebook_similarity(codebooks, input_ids):
    similarities = []

    for codebook in codebooks:
        sim = jaccard_similarity(list(codebook), list(input_ids.reshape(-1)))
        similarities.append(sim)
    return similarities
