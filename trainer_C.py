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

sess = tf.Session()
from keras import backend as K

K.set_session(sess)
from keras.layers import Dense, Conv2D, Reshape, Flatten, Dropout
from keras.metrics import categorical_accuracy as accuracy2

img = tf.placeholder(tf.float32, shape=[None, 6912], name="Image")


"""Keras layers"""

x = Dense(6912, activation='relu',  kernel_initializer='random_uniform',bias_initializer='zeros')(img)
x = Dropout(0.20)(x)
x = Reshape((48, 48, 3))(x)
x = Conv2D(32, (3,3), activation='relu',  kernel_initializer='random_uniform',bias_initializer='zeros')(x)
x = Conv2D(64, (3,3), activation='relu',  kernel_initializer='random_uniform',bias_initializer='zeros')(x)
x = Conv2D(32, (3,3), activation='relu',  kernel_initializer='random_uniform',bias_initializer='zeros')(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(1000, activation='relu',  kernel_initializer='random_uniform',bias_initializer='zeros')(x)
preds = Dense(10, activation='softmax')(x)

labels = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass")

from keras.objectives import categorical_crossentropy

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
# loss = categorical_crossentropy(labels, preds)
train_step = tf.train.AdamOptimizer(0.00005).minimize(loss)
	# Bij AdamOptimizer is een zeer kleine learning rate gebruikelijk
	# AdamOptimizer gaat zijn learningRate zelf aanpassen dus dit zelf doen is niet echt nodig

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

	loopAmount = 5000

	accuracyArray = [0, 0, 0]
	for x in range(loopAmount):
		training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
		training_set_image /= 255
		# print(training_set_image[0])
		sess.run(train_step, feed_dict={img: training_set_image, labels: training_set_class, K.learning_phase():1})
		if x % 100 == 0:
			# print(training_set_name[1])
			evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filename = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			evaluation_set_image /= 255
			accuracy = sess.run(accuracy_value, feed_dict={img: evaluation_set_image, labels: evaluation_set_class,  K.learning_phase():0})
			accuracyArray.append(accuracy)
			print(str(x) + ": " + str(accuracy))
		if x % 1000 == 0:
			print('%s%s ======= Iteration %s of %s | %s done ======= %s' % (fg('white'), bg('red'), str(x), str(loopAmount), str("{0:.0f}%".format((x/loopAmount) * 100)), attr('reset')))

	input("Press Enter to continue...")

	evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filenames = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN

	coord.request_stop()
	coord.join(threads)
