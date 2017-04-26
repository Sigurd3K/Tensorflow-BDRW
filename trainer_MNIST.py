"""
Full Tensorflow code with keras to manage layers
https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

https://github.com/GINK03/KindleReferencedIndexScore/blob/debdc8e0b68459b0d6870a52e5f6ac58ee1c1d25/keras-text-classification/model.py
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
from keras.models import Model
from keras.layers import Dense, Conv2D, Reshape, Flatten, Dropout, MaxPooling2D, InputLayer, Input
from keras.metrics import categorical_accuracy as accuracy2
from keras.models import load_model
import os


img = tf.placeholder(tf.float32, shape=[None, 784], name="Image")


"""Keras layers"""

inputs = Input(tensor=img)

# x = Dense(784)(img)
# x = InputLayer(input_tensor=inputs, input_shape=(None, 784))
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
	# Je kan ook gewoon preds ingeven ipv van model.output maar ik wil alle getrained dingen kunnen opslagen

LearningRate = 0.0001
train_step = tf.train.AdamOptimizer(LearningRate).minimize(loss)
	# Bij AdamOptimizer is een zeer kleine learning rate gebruikelijk
	# Is een optimizer met momentum
	# AdamOptimizer gaat zijn learningRate zelf aanpassen dus dit zelf doen is niet echt nodig

"""Calculate Accuracy"""
accuracy_value = accuracy2(labels, preds)
accuracy_value = tf.reduce_mean(tf.cast(accuracy_value, tf.float32))

# score = preds.evaluate(img, labels, verbose=0)

# Model Saver ------------------------------

def ModelSaver():
	K.set_learning_phase(0)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# Model to HDF5
	model.save("model.h5", overwrite=True)
	model.save_weights("modelWeights.h5", overwrite=True)
	print("Saved model to disk")
# -------------------------------------------

"""Start of Tensorflow Session"""

with sess.as_default():

	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	looper = 0

	printedTest = False

	loopAmount = 60000
	TestImages = mnist.test.images
	TestLabels = mnist.test.labels

	TestImgLen = len(TestImages)
	TestBatchSize = 100
	meanAccuracy = []

	accuracyArray = []
	for x in range(loopAmount):
		batch = mnist.train.next_batch(100)
		# training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
		# print(training_set_image[0])
		sess.run(train_step, feed_dict={img: batch[0], labels: batch[1], K.learning_phase():1})
		if x % 100 == 0:
			for amount in (range(0,TestImgLen,TestBatchSize)): # Evaluate accuracy on whole MNIST_Test_images set.
				start = amount
				end = amount + TestBatchSize
				accuracy = sess.run(accuracy_value, feed_dict={img: TestImages[start:end], labels: TestLabels[start:end], K.learning_phase():0})
				accuracyArray.append(accuracy)
			meanAccuracy.append(np.mean(accuracyArray)) # The mean accuracy (Accuracy on the whole batch
			print(str(x) + ": " + str(meanAccuracy[-1:])) # Print the mean accuracy (Accuracy on the whole batch and the previous one)
			if len(meanAccuracy) > 10 and x%1000 == 0:
				if (mean(meanAccuracy[-4:]) - mean(meanAccuracy[-8:-4])) <=0.01:
					LearningRate *= 0.9
					print("LearningRate is now " + str(LearningRate))
		if x % 1000 == 0:
			print('%s%s ======= Iteration %s of %s | %s done ======= %s' % (fg('white'), bg('red'), str(x), str(loopAmount), str("{0:.0f}%".format((x/loopAmount) * 100)), attr('reset')))
	input("Press Enter to continue...")
	# ModelSaver()

	saver = tf.train.Saver()
	save_path = saver.save(sess, "tmp/model.ckpt")
	print("Model saved in file: %s" % save_path)

	# evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filenames = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN

	coord.request_stop()
	coord.join(threads)
# # Laatste stuk proberen: https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
