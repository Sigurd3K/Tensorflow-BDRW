"""
Full Tensorflow code with keras to manage layers
https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
"""

import numpy as np
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
from keras.models import Model
from keras.layers import Dense, Conv2D, Reshape, Flatten, Dropout, MaxPooling2D, InputLayer, Input
from keras.metrics import categorical_accuracy as accuracy2
from keras.models import load_model
from keras.models import model_from_json

import os


img = tf.placeholder(tf.float32, shape=[None, 784], name="Image")
labels = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass")


##################################
# ---- LOAD MODEL AND WEIGHTS --##

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
loaded_model = load_model('model.h5')
# load_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# K.set_learning_phase(0)

## evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# load weights into new model
# loaded_model.load_weights("model.h5")
print("Loaded model from disk")

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(loaded_model.output)

#
# def scoreFunc(img, labels):
	# print(type(img))
	# print(type(labels))
	# evaluational = loaded_model.evaluate(img, labels, batch_size = 50)
	# return np.mean(xx[1:3])
	# return loaded_model.evaluate(img, labels, batch_size = 50)
	# accuracy = sess.run(, feed_dict={img: TestImages[start:end], labels: TestLabels[start:end], K.learning_phase():0})
	# return np.random.rand(10)

#
# score = tf.py_func(scoreFunc, [img, labels], tf.float64)

# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# ---- LOAD MODEL AND WEIGHTS --##
##################################


# accuracy_value = accuracy2(labels,sdsdsdsdsdsd)
# accuracy_value = tf.reduce_mean(tf.cast(accuracy_value, tf.float32))

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

"""Start of Tensorflow Session"""

K.set_session(sess)

batch = []

with sess.as_default():
	# K.set_session(sess)

	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()


	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	batch = mnist.train.next_batch(100)

	loaded_model.load_weights("modelWeights.h5")
	sess.run(c, feed_dict={K.learning_phase():0})
	# score2 = sess.run(score, feed_dict={img: batch[0], labels: batch[1]})
	# evaluational = loaded_model.evaluate(batch[0], batch[1], batch_size = 50)
	predictions = loaded_model.predict(batch[0])
	preds2 = np.argmax(predictions, axis=1)
	print(preds2)
	print(np.argmax(batch[1], axis=1))
		# Predict classes is voor de sequential niet voor de functional API
	predictions = loaded_model.evaluate(batch[0], batch[1], verbose=0, batch_size=50)
	print(predictions[1])
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], predictions[1]*100))
	# preds2 = np.argmax(predictions, axis=1)
	# print(length(batch[]))
	# print(preds2)
	# print(len(preds2))
	# print(batch[1])
	# print(np.argmax(predictions, axis=1))
	# print(predictions)

	# print(evaluational[0])
	# print(evaluational[1])
	# print(size(evaluational))
	# print(score2)
	# score2 = sess.run(score, feed_dict={img: batch[0], labels: batch[1], K.learning_phase():0})

	# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score2[1]*100))

	# accuracy = sess.run(accuracy_value, feed_dict={img: batch[0], labels: batch[1], K.learning_phase():0})
	# print( "Accuracy: " + str(accuracy))


coord.request_stop()
coord.join(threads)

predictions = loaded_model.predict(batch[0])
preds2 = np.argmax(predictions, axis=1)
print(preds2)
print(np.argmax(batch[1], axis=1))
	# Predict classes is voor de sequential niet voor de functional API
predictions = loaded_model.evaluate(batch[0], batch[1], verbose=0, batch_size=50)
print(predictions[1])
print("%s: %.2f%%" % (loaded_model.metrics_names[1], predictions[1]*100))

# Laatste stuk proberen: https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
