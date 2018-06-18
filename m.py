import os
import matplotlib
matplotlib.use('Agg')
path_for_mnist_metadata = os.path.join(os.getcwd(), "labels_1024.tsv")
path_for_mnist_sprites = os.path.join(os.getcwd(), "sprite_1024.png")
embedding_size = 1024
LOG_DIR = 'minimalsample3'
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector
import argparse
import sys
FLAGS = None
sess = tf.Session()
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(1000)
train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)


embedding_var = tf.Variable(batch_xs, name="NAME_TO_VISUALISE_VARIABLE")
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Specify where you find the metadata
embedding.metadata_path = path_for_mnist_metadata
# Specify where you find the sprite (we will create this later)
embedding.sprite.image_path = path_for_mnist_sprites #'mnistdigits.png'
embedding.sprite.single_image_dim.extend([28,28])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(train_writer, config)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))


    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


to_visualise = batch_xs
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')

with open(path_for_mnist_metadata,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(batch_ys):
        f.write("%s\t%s\n" % (index,label))

def variable_summaries(var, name):

    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('sttdev/' + name, stddev)
      tf.summary.scalar('max/' + name, tf.reduce_max(var))
      tf.summary.scalar('min/' + name, tf.reduce_min(var))
      tf.summary.histogram(name, var)
        


with tf.variable_scope("input"):
    x = tf.placeholder("float", [None, 784], name="input_x")
    y_ = tf.placeholder("float",[None,10],name="input_label")

with tf.variable_scope("first-nn-layer"):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x,W) + b ,name="our_predict")
    tf.summary.histogram("W", W)
    tf.summary.histogram("b", b)

with tf.variable_scope("loss"):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    variable_summaries(cross_entropy,"cross_entropy")
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
variable_summaries(accuracy, "accuracy")


init = tf.global_variables_initializer()


merged = tf.summary.merge_all()


sess.run(init)

for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  summary = sess.run(merged, feed_dict={x:batch_xs,y_:batch_ys})
  train_writer.add_summary(summary, i)

print str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100) + "%"
