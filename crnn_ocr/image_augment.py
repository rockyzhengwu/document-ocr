#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

from imgaug import augmenters as iaa

seq = iaa.SomeOf((1, 4), [
  iaa.Salt(p=(0.1, 0.2)),
  iaa.GaussianBlur(sigma=(0, 0.5)) ,
  iaa.CoarseDropout(p=(0.02, 0.1), size_percent=(0.2, 0.3)),
  iaa.JpegCompression(compression=(50,80)),
])



def augment_images(images):
  images_aug = seq(images=images)
  return images_aug 
    

