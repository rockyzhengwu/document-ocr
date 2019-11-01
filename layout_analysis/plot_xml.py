#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

from bs4 import BeautifulSoup
import random
import cv2
import os

def plot_element(image, item, page_height, page_width, color):
  image_height, image_width = image.shape[:2]
  width_ratio =  image_width/ page_width 
  height_ratio = image_height / page_height
  left = int(item.get("left")) * width_ratio
  top = int(item.get("top")) * height_ratio
  width = int(item.get("width")) * width_ratio
  height = int(item.get("height")) * height_ratio
  pt1 = (int(left), int(top))
  pt2 = (int(left + width), int(top + height))
  image=cv2.rectangle(image, pt1=pt1, pt2=pt2, color=color, thickness=2) 
  return image


def plot_from_path(xml_path, image_path):
  with open(xml_path) as f:
    soup = BeautifulSoup(f.read(), 'xml')
  page = soup.find('page')
  image = cv2.imread(image_path)
  out_image_path = os.path.join('./test_out', page.get("number") + ".png")
  plot_one(page, image, out_image_path)

def plot_one(page, image, out_image_path):
  h,w = image.shape[:2]
  page_height = int(page.get("height"))
  page_width = int(page.get("width"))
  image_list = page.find_all('image')
  text_list = page.find_all('text')
  for item in image_list:
    image = plot_element(image, item, page_height, page_width, (125, 0, 255))
  for item in text_list:
    image = plot_element(image, item, page_height, page_width, (255, 255, 0))
  cv2.imwrite(out_image_path, image)


def run(xml_path, out_path):
  with open(xml_path, encoding='utf-8', errors='ignore') as f:
    xml_data = f.read()
  soup = BeautifulSoup(xml_data, 'xml')
  pages = soup.find_all('page')
  for page in pages:
    image_path = page.get('image_path')
    print(image_path)
    image= cv2.imread(image_path)
    out_image_path = os.path.join(out_path, page.get("number") + ".png")
    plot_one(page, image, out_image_path)


def plot_page():
  xml_path_list = []
  for top, dirs, files in os.walk('/data/zhengwu_workspace/document_text_line/layout_xml_anno/year_report'):
    for name in files:
      xml_path = os.path.join(top, name)
      xml_path_list.append(xml_path)

  random.shuffle(xml_path_list)
  for xml_path in xml_path_list[:100]:
    print(xml_path)
    with open(xml_path) as f:
      xml_data = f.read()
    soup = BeautifulSoup(xml_data, 'xml')
    page = soup.find('page')
    out_image_path = xml_path.split("/")[-1].replace("xml", "png")
    out_image_path = os.path.join('./test_out', out_image_path)
    print(out_image_path)
    plot_one(page, out_image_path)

if __name__ == "__main__":
  xml_path = "/home/zhengwu/data/book_layout/part_2/314560.xml"
  image_path = "./dataset/input.png"
  plot_from_path(xml_path, image_path)
