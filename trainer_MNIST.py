"""
Full Tensorflow code with keras to manage layers
https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
"""

import tensorflow as tf
import math
from colored import fg, bg, attr
import fileReader as fR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()
from keras import backend as K

K.set_session(sess)
from keras.layers import Dense, Conv2D, Reshape, Flatten, Dropout, MaxPooling2D
from keras.metrics import categorical_accuracy as accuracy2

img = tf.placeholder(tf.float32, shape=[None, 784], name="Image")


"""Keras layers"""

x = Dense(784)(img)
x = Reshape((28, 28, 1))(x)
x = Conv2D(32, (3,3), activation='relu')(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.20)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

labels = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass")

from keras.objectives import categorical_crossentropy

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
# loss = categorical_crossentropy(labels, preds)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

"""Calculate Accuracy"""
accuracy_value = accuracy2(labels, preds)
accuracy_value = tf.reduce_mean(tf.cast(accuracy_value, tf.float32))

# score = preds.evaluate(img, labels, verbose=0)

"""Start of Tensorflow Session"""

with sess.as_default():
	# K.set_session(sess)

	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	looper = 0

	printedTest = False

	for x in range(loopAmount):
	input("Press Enter to continue...")


	coord.request_stop()
	coord.join(threads)
