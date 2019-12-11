#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wu.zheng midday.me

import cv2

from albumentations import (
  HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
  Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
  IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
  IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomBrightness
)


def image_aug(image, p=1.0):
  def strong_aug(p=p):
    return Compose([
      OneOf([
        IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.1 * 255)),
        GaussNoise(var_limit=(10.0, 100.0)),
      ], p=0.5),
      OneOf([
        MotionBlur(p=0.2),
        MedianBlur(blur_limit=7, p=0.1),
        Blur(blur_limit=7, p=0.1),
      ], p=0.5),
      OneOf([
        IAASharpen(p=1.0),
        IAAEmboss(p=1.0),
        CLAHE(clip_limit=2, p=1.0),
        RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=1.0),
      ], p=0.5),
      HueSaturationValue(p=0.5),
    ], p=p)

  augmentation = strong_aug(p=p)
  augmented = augmentation(image=image)
  aug_image = augmented['image']
  return aug_image


