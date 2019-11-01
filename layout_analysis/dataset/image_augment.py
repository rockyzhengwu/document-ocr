#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random

seq = iaa.SomeOf((1, 4), [
    iaa.Salt(p=(0.2, 0.4)),
    iaa.GaussianBlur(sigma=(0, 2.0)) ,
    iaa.CoarseDropout(p=(0.02, 0.1), size_percent=(0.2, 0.6)),
    iaa.JpegCompression(compression=(30,50)),
])


def augment_with_segmap(image, segmap, num_classes):
  if random.random() < 0.3:
    return image, segmap
  segmap = SegmentationMapsOnImage(segmap, shape=image.shape, nb_classes=num_classes)
  image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
  return image_aug, None


