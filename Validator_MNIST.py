"""
	Loads the model trained by the MNIST Trainer and runs it to validate
	COMPLETELY FUNCTIONAL
"""

import tensorflow as tf
import math
from colored import fg, bg, attr
# import fileReader as fR

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()
from keras import backend as K

K.set_session(sess)

from keras.models import Model
from keras.layers import Dense, Conv2D, Reshape, Flatten, Dropout, MaxPooling2D, InputLayer, Input
from keras.metrics import categorical_accuracy as accuracy2
from keras.models import load_model
import os

img = tf.placeholder(tf.float32, shape=[None, 784], name="Image")

"""Keras layers"""

inputs = Input(tensor=img)

x = Dense(784)(inputs)
x = Reshape((28, 28, 1))(x)
x = Conv2D(32, (3,3), activation='relu')(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

model = Model(inputs=img, outputs= preds)

labels = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass")

from keras.objectives import categorical_crossentropy

loss = tf.reduce_mean(categorical_crossentropy(labels, model.output))
# In plats van een Keras optimizer kunnen we bijvoorbeeld gewoon een Tensorflow optimizer op de Keras layers gebruiken
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

"""Calculate Accuracy"""
accuracy_value = accuracy2(labels, preds)
accuracy_value = tf.reduce_mean(tf.cast(accuracy_value, tf.float32))

"""Start of Tensorflow Session"""

with sess.as_default():
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	saver = tf.train.Saver()
	saver.restore(sess, "tmp/model.ckpt")

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	looper = 0

	TestImages = mnist.test.images
	TestLabels = mnist.test.labels

	TestImgLen = len(TestImages)
	print(str(TestImgLen))

	TestBatchSize = 100

	meanAccuracy = []
	accuracyArray = []
	for amount in (range(0,TestImgLen,TestBatchSize)): # Evaluate accuracy on whole MNIST_Test_images set.
		start = amount
		end = amount + TestBatchSize
		accuracy = sess.run(accuracy_value, feed_dict={img: TestImages[start:end], labels: TestLabels[start:end], K.learning_phase():0})
		accuracyArray.append(accuracy)
	meanAccuracy.append(np.mean(accuracyArray))
	print(str("Accuracy :") + str(meanAccuracy[-1:])) # The mean accuracy (Accuracy on the whole batch)
	input("Press Enter to continue...")

	coord.request_stop()
	coord.join(threads)
