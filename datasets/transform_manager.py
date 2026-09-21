import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image


class PairedConsistencyTransform:
    """Return two photometrically different views in one spatial frame.

    The two views share exactly the same random crop and horizontal flip.
    Consequently, attention-map token ``(row, column)`` refers to the same
    image region in both views and their alignment distributions can be
    compared without an additional geometric warp.
    """

    def __init__(self, transform_type, image_size=84):
        self.transform_type = transform_type
        self.image_size = image_size
        self.primary_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4
        )
        self.augmented_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _shared_geometry(self, image):
        if self.transform_type == 0:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            )
            image = TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                (self.image_size, self.image_size),
            )
        elif self.transform_type == 1:
            image = TF.pad(image, 8)
            top, left, height, width = transforms.RandomCrop.get_params(
                image,
                output_size=(self.image_size, self.image_size),
            )
            image = TF.crop(image, top, left, height, width)
        else:
            raise Exception('transform_type must be specified during training!')

        # Flip both views together.  An independently sampled flip would put
        # the two alignment maps into different spatial coordinate systems.
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)

        return image

    def __call__(self, image):
        image = self._shared_geometry(image)
        primary = self.normalize(self.primary_jitter(image))
        augmented = self.normalize(self.augmented_jitter(image))
        return primary, augmented

def get_transform(is_training=None,transform_type=None,pre=None):

    if is_training and pre:
        raise Exception('is_training and pre cannot be specified as True at the same time')

    if transform_type and pre:
        raise Exception('transform_type and pre cannot be specified as True at the same time')

    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean,std=std)
                                    ])

    if is_training:

        if transform_type == 0:
            size_transform = transforms.RandomResizedCrop(84)
        elif transform_type == 1:
            size_transform = transforms.RandomCrop(84,padding=8)
        else:
            raise Exception('transform_type must be specified during training!')
        
        train_transform = transforms.Compose([size_transform,
                                            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                            transforms.RandomHorizontalFlip(),
                                            normalize
                                            ])
        return train_transform
    
    elif pre:
        return normalize
    
    else:
        
        if transform_type == 0:
            size_transform = transforms.Compose([transforms.Resize(92),
                                                transforms.CenterCrop(84)])
        elif transform_type == 1:
            size_transform = transforms.Compose([transforms.Resize([92,92]),
                                                transforms.CenterCrop(84)])
        elif transform_type == 2:
            # for tiered-imagenet and (tiered) meta-inat where val/test images are already 84x84
            return normalize

        else:
            raise Exception('transform_type must be specified during inference if not using pre!')
        
        eval_transform = transforms.Compose([size_transform,normalize])
        return eval_transform


def get_paired_consistency_transform(transform_type):
    """Build paired training views for alignment-consistency regularization."""
    return PairedConsistencyTransform(transform_type=transform_type)
