"""
Cars Detection
"""

import scipy.io
import matplotlib.image as mpimg
from PIL import Image

train_annon = scipy.io.loadmat('/Users/prasanna/Downloads/devkit/cars_train_annos.mat')

a=train_annon['annotations']

class_arr = []
root='/Users/prasanna/Downloads/cars_train'
import os
image_names = os.listdir(root)
count = 0
box_arr = []
for item in image_names:
    file_path = root+"/"+item
    # image = mpimg.imread(file_path).astype(float)
    #image.shape
    img = Image.open(file_path)
    class_arr.append(a[0][count][4][0][0])
    box = (a[0][count][0][0][0],a[0][count][1][0][0],a[0][count][2][0][0],a[0][count][3][0][0])
    box_arr.append(box)
    count += 1
    img = img.convert('L')
    region = img.crop(box)
    region = region.resize((28, 28), Image.ANTIALIAS)
    save_folder = '/Users/prasanna/Downloads/cars_train_gray/'
    region.save(save_folder + item)


import matplotlib.image as mpimg

image_features_arr = []
root='/Users/prasanna/Downloads/cars_train_gray'
for item in image_names:
    file_path = root+"/"+item
    image = mpimg.imread(file_path).astype(float)
    image.resize((28, 28, 1))
    image_features_arr.append(image)



import numpy as np
image_features_numpy_arr = np.array(image_features_arr)
class_numpy_arr = np.array(class_arr)
class_unique = np.sort(np.unique(class_numpy_arr))
class_label = []
len_class_unique=len(class_unique)
for item in class_arr:
    temp = [0]*len_class_unique
    temp[item-1] = 1
    class_label.append(temp)

class_numpy_label = np.array(class_label)
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

total_data = image_features_numpy_arr.shape[0]
image_features_numpy_arr_train = image_features_numpy_arr[0:total_data*95/100]
class_numpy_label_train = class_numpy_label[0:total_data*95/100]
image_features_numpy_arr_test = image_features_numpy_arr[total_data*95/100+1:]
class_numpy_label_test = class_numpy_label[total_data*95/100+1:]
car_data_train = DataSet(images=image_features_numpy_arr_train,labels=class_numpy_label_train)
car_data_test = DataSet(images=image_features_numpy_arr_test,labels=class_numpy_label_test)



import tensorflow as tf
x = tf.placeholder("float", shape=[None, 784])
#
y_ = tf.placeholder("float", shape=[None, 196])
x_image = tf.reshape(x, [-1,28,28,1])
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#
W_fc2 = weight_variable([1024, 196])
b_fc2 = bias_variable([196])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(200):
  batch = car_data_train.next_batch(50)
  if i%10 == 0:
        train_accuracy = sess.run( accuracy, feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print("test accuracy %g"% sess.run(accuracy, feed_dict={
    x: car_data_test.images, y_: car_data_test.labels, keep_prob: 1.0}))
