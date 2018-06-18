
---
title: Tensorflow-学习笔记
date: 2018-04-21 16:14:27
tags:
---
最近学习了一下Google的开源机器学习框架，简单写一下自己对于MNIST学习过程的理解

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
```

这里的几行分别包含了Tensorflow的库和学习所需要的包（并非重点）

```python
FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```

这里将数据下载到本地的/tmp/tensorflow下以便机器学习的时候使用

```python
x = tf.placeholder("float",[None,784])
W = tf.Variable(tf.zeros([784,10]))
```

首先以占位符的形式定义了x这一张量，x拥有两个维度，第一个维度我们设为None，因为我们需要读取图片，并将其的顺序放在第一个维度上，第二个维度拥有784个数，也就是789个像素，这样就可以完成数据的读取。

之后我们定义了W这一权重值，这里有可能会理解错。w在第一个维度上有784个数字，也就是对应每一位都有对应的权重，第二个维度是分类。我们要将0~9这些数字分成十类分别进行加权求和，所以定义这样一个张量。

```python
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b )
y_ = tf.placeholder("float",[None,10])

```
下面我们又定义了b作为偏置或者称为修正值，类似一次函数。

之后，我们让y作为运算结果，让x和W进行矩阵相乘，最后分别加上修正值b。再对计算后的结果进行Softmax回归运算，这样我们就可以得到输入图片对应每一个类的可能性。

```python
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(50)

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100) + "%"

```

下面我们要求使用梯度下降算法降低loss（交叉熵）并且每次读取50个数字并循环10000次。
最后输出评估结果，大约在92%左右
