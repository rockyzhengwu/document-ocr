#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import os

import tensorflow as tf
import json
import numpy as np
import time
import densenet
import data_generator
import argparse

parser = argparse.ArgumentParser(description="SingleWord Reconition")
parser.add_argument('--test_image_list', help='test_data_dir')
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--batch_size', default=128, help='batch size', type=int)
parser.add_argument('--num_class', help='num of class', type=int)
parser.add_argument('--export', type=bool, help='whether to export model', dest='export')
parser.add_argument('--export_dir', help='export saved model dir')


def export_model(sess, model):
  tensor_image_input_info = tf.saved_model.utils.build_tensor_info(model.images)
  tensor_prob_output_info = tf.saved_model.utils.build_tensor_info(model.predict_prob)
  tensor_prediction_output_info = tf.saved_model.utils.build_tensor_info(model.prediction)
  signature = tf.saved_model.signature_def_utils.build_signature_def(
          inputs = {"images": tensor_image_input_info},
          outputs = {"prob": tensor_prob_output_info, "prediction": tensor_prediction_output_info}
  )
  ex_dir = str(int(time.time()))
  export_dir = os.path.join(args.export_dir, ex_dir)
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  builder.add_meta_graph_and_variables(
          sess=sess,
          tags=[tf.saved_model.tag_constants.SERVING], 
          signature_def_map={"predict": signature})
  builder.save()
  print("export model at %s"%(export_dir))

def load_vocab():
  data = json.load(open("gbk_eng.json"))
  data_reverse = {v:k for k, v in data.items()}
  return data_reverse

def main():
  save_path = tf.train.latest_checkpoint(args.model_dir)
  model = densenet.DenseNet(1, args.num_class, mode='test')
  saver = tf.train.Saver()
  id_to_word = load_vocab()

  with tf.Session() as sess:
    saver.restore(sess=sess, save_path=save_path)
    if args.export:
      export_model(sess, model)
      exit(0)

    print("load model from %s"%(save_path))
    counter = 0
    right_counter = 0
    for batch_data in data_generator.get_batch(args.test_image_list, batch_size=1, mode='test', workers=1, max_queue_size=12):
      image = np.array(batch_data[0])
      label = batch_data[1]
      image_path = batch_data[2]
      feed_dict = {model.images: image}
      prediction, predict_prob = sess.run([model.prediction, model.predict_prob], feed_dict=feed_dict)
      predict_id = prediction[0]
      predict_label = id_to_word[predict_id]
      predict_prob = predict_prob[0][predict_id]
      true_label = id_to_word[label[0]]
      print("image_path: %s, true_id: %d, true_label: %s, predict_label: %s, predict_prob: %f"%(
        image_path, label[0], true_label ,predict_label, predict_prob))

      if true_label == predict_label :
        right_counter += 1
      counter += 1
      if counter > 100:
        break
    print("acc : %f"%(1.0 * right_counter / counter ))

if __name__ == "__main__":
  args = parser.parse_args()
  print(args)
  main()



