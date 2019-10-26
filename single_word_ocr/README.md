## 单字识别

可以使用[合成工具](https://github.com/rockyzhengwu/synthtext)合成数据


模型采用 densenet 


## 使用



1. 准备数据生成image_list 文件格式如下, 准备对应的字典文件
```
/data/9916/Z01.png 9916
/data/9916/Z02.png 9916
```

2. 训练数据

训练不需要生成tfrecord 文件

num_class: 字典文件参数

python train.py --train_image_list <image_list> --num_class <num_class> --checkpoint_path <checkpont_path>


其他参数见代码

3. 测试

python eval.py 具体参数详见代码



