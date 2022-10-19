import os
import json
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve


class SimpleImageHarmonizationMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, odgt, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform
        self.parse_input_list(odgt)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, idx):
        this_record = self.list_sample[idx]
        image_path = os.path.join(self.root, this_record['fpath_img'])
        mask_path = os.path.join(self.root, this_record['fpath_segm'])

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path).convert('L'))
        mask = self._preprocess_mask(trimap)

        _, tail = os.path.split(this_record['fpath_img'])
        sample = dict(image=image, mask=mask, trimap=trimap, name=tail)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask > 0.0] = 1.0
        return mask


class ImageHarmonizationMaskDataset(SimpleImageHarmonizationMaskDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # image = sample["image"]
        # mask = sample["mask"]
        # trimap = sample["trimap"]

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

