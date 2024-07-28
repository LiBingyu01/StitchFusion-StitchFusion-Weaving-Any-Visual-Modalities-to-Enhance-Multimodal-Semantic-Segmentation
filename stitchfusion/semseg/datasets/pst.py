import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation

class PST(Dataset):
    """
    num_classes: 5
    """
    CLASSES = ["Background", "Fire-Extinguisher", "Backpack", "Hand-Drill", "Survivor"]

    PALETTE = torch.tensor([[0, 0, 0],
            [100, 40, 40],
            [55, 90, 80],
            [220, 20, 60],
            [153, 153, 153]])
    
    def __init__(self, root: str = 'data/PST900', split: str = 'train', transform = None, modals = ['img', 'thermal'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        if split == 'val':
            split = 'test'
        self.files = sorted(glob.glob(os.path.join(*[root, split, 'rgb', '*.png'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        # --- split as case
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = self.files[index].split("/")[-1].split(".")[0]
        rgb = str(self.files[index])
        thermal = rgb.replace('/rgb', '/thermal')
        lbl_path = rgb.replace('/rgb', '/labels')

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        H, W = sample['img'].shape[1:]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_img(thermal)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        # label[label==255] = 0
        # label -= 1
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        # return sample, label, item_name
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)


if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:

        trainset = DELIVER(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))