import os
from PIL import Image
import numpy as np

import torch
import glob
from torch.utils.data import Dataset
import pycocotools.mask as coco_mask
from pycocotools.coco import COCO


def remove_leading_zeros(input_string):
    if not input_string:
        return input_string

    non_zero_index = next((i for i, digit in enumerate(input_string) if digit != '0'), None)

    return int(input_string[non_zero_index:]) if non_zero_index is not None else input_string


class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, seg_num=False, pose_num=False):
        self.root = root
        self.coco = COCO(annFile)
        # self.ids = sorted(list(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform
        self.seg_num = seg_num
        self.pose_num = pose_num
        if self.seg_num:
            imgs_path = glob.glob(os.path.join(root,'*.jpg'))[:50000]
        elif self.pose_num:
            imgs_path = glob.glob(os.path.join(root,'*.jpg'))[:30000]
        else:
            imgs_path = glob.glob(os.path.join(root,'*.jpg'))

        self.ids = [remove_leading_zeros(os.path.basename(path)[:-4]) for path in imgs_path]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))

        if self.transform is not None:
            img_tr = self.transform(img)
        else:
            img_tr = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_tr, img, target, img_id

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


# Add colorful segmentation mask on the image.
def convert_anns_to_mask_seg(img_sample):
    # Process on cpu
    device = "cuda"
    img,_, anns, _ = img_sample
    img = Image.fromarray(img).convert('RGB')
    width, height, filename = img.size[0], img.size[1], str(anns[0]["image_id"]).zfill(12)
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask_img = torch.zeros((height, width, 3), device=device)
    one_img = torch.ones((height, width, 3), device=device)

    for ann in sorted_anns:
        # Polygons to mask
        mask = convert_coco_poly_to_mask(ann['segmentation'], height, width)
        mask = torch.tensor(mask, device=device)
        mask = torch.repeat_interleave(mask.squeeze(dim=0).unsqueeze(dim=2), repeats=3, dim=2)
        color_mask = torch.rand(3, device=device) * one_img
        mask_img += color_mask * mask

        del mask, color_mask
        # torch.cuda.empty_cache()
    mask_img_npy = mask_img.cpu().numpy()
    mask_img_npy = (255 * mask_img_npy).astype(np.uint8)

    del mask_img, one_img
    # torch.cuda.empty_cache()

    return mask_img_npy, filename


def convert_coco_poly_to_mask(polygons, height, width):
    masks = []
    rles = coco_mask.frPyObjects(polygons, height, width)
    mask = coco_mask.decode(rles)
    if len(mask.shape) < 3:
        mask = mask[..., None]
    mask = torch.as_tensor(mask, dtype=torch.uint8)
    mask = mask.any(dim=2)
    masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


if __name__ == "__main__":
    # Save gt image path
    save_path = "coco_seg/gt_prompts/"
    # GT Annotation path 
    anno_file_path = "../dataset/annotations/instances_train2017.json"
    # GT image path
    img_file_path = '../dataset/coco/train2017/'

    datasets = CocoDetection(img_file_path, anno_file_path)
    print(datasets)

    for dataset in datasets:
        if len(dataset[2]) == 0:
            continue
        else:
            color_masked_img, img_name = convert_anns_to_mask_seg(dataset)
            print("img_name:", img_name)
            im = Image.fromarray(color_masked_img)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            im.save(os.path.join(save_path, img_name + ".jpg"))
