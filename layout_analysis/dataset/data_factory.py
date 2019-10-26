#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

from dataset import book_data

BATCH_SIZE = 4
MODE = 'train'
WORKERS = 1
MAX_QUEUE_SIZE = 16

DEFAULT_PARAM = {
    "book": {
      "train_data_dir": "train.list",
      "image_dir":"",
      "test_data_dir": "test.list",
      "num_class": 3,
      "check_points_path":"/data/zhengwu_workspace/ocr/models/book_unet/image_txt_aux",
      "data_fn": book_data
      },
    }

def get_data_iterator(name, **kargs):
  default_param = DEFAULT_PARAM.get(name)
  if default_param is None:
    raise Exception("data %s is not defained "%(name))
  batch_size = kargs.get("batch_size", BATCH_SIZE)
  mode = kargs.get("mode", MODE)
  workers = kargs.get("workers", WORKERS)
  max_queue_size = kargs.get("max_queue_size", MAX_QUEUE_SIZE)
  data_fn = default_param.get('data_fn')
  if mode=='train':
    data_dir = default_param.get('train_data_dir')
  else:
    data_dir = default_param.get('test_data_dir')
  data_iterator = data_fn.get_batch(data_dir, image_dir, batch_size, mode=mode, workers=workers, max_queue_size=max_queue_size)
  return data_iterator

def get_data_config(name):
  default_param = DEFAULT_PARAM.get(name)
  return default_param

if __name__ == "__main__":
  for batch_data in get_data_iterator('book'):
    if batch_data is None:
      continue
    print(batch_data[0][0].shape)
    #exit(0)

