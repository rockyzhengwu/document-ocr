## 文本行识别

CRNN + RNN + CTC 实现

## 使用

1. 准备标注数据文件，文件路径最好是绝对路径,路径和文本之间用空格隔开


```
/data/9b9723f0-f7e4-49b4-bc95-28cd1cdd28e0.png 游的片曲「Come Home! Princess」是
/data/4ed93c5d-b0f6-4232-a16a-78bdd5296a08.png 有8个公交港湾，留5个大的出入口潘多脂
/data/1d588889-e28e-4b33-8705-b10865785efe.png 摩哥大
/data/334c4175-d25e-4d61-b5eb-576f8983a0fd.png 甸，中国古代官名，于周礼》中，主管

```

字典数据用 json 存放格式如下,如果有在字典之外的符号统一用`<UNK>`代替


```
{
  "<UNK>": 0,
  "天":1,
  "文":2,
}

```

2. 创建 tfrecord 文件

- image_list : 是上面准备的数据文件
- data_dir：存放tf_record 路径
- vocab_file: 是准备的词典文件


```
python ./create_tfrecord.py --image_list ${LABELS_FILE} --vocab_file {vocab.json} --data_dir ${TF_RECRD_DIRS} --max_seq_length ${MAX_SEQ_LENGTH} --channel_size ${CHANNEL_SIZE}
```

代码会使用多线程创建多份 train_tfrecord 文件，具体其他参数可以自行修改代码

```
start_create_process(train_anno_lines, 100, 10, 'train')
start_create_process(validation_anno_lines, 10, 10,  'validation')
start_create_process(test_anno_lines, 10, 10,  'test')

```

3. 训练


```
python  train.py --data_dir ${TF_RECRD_DIRS} --model_dir ${MODEL_DIR} --max_seq_length ${MAX_SEQ_LENGTH} --channel_size ${CHANNEL_SIZE}

```


4. 测试


```
python ./eval.py --max_seq_length ${MAX_SEQ_LENGTH} --channel_size ${CHANNEL_SIZE} --model_dir ${MODEL_DIR} --image_list ${LABELS_FILE} --image_dir ${IMAGE_DIR}

```

直接使用image_list 格式的数据作为输入，方面查看 bad case,如果需要读入 tfrecord 批量测试需要自行实现相关代码

增加 ```export```参数可以导出模型 使用 ```load_saved_model.py```的样例代码读取 saved model



