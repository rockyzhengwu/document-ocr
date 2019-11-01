#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

from dataset import book_data


DATA_GENERATOR = {
  "book": book_data
}

def get_data(name):
  data_generator_fn = DATA_GENERATOR.get(name)
  if data_generator_fn is None:
    print("data %s not exists")
  return data_generator_fn

