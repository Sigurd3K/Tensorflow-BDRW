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

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

"""Calculate Accuracy"""
accuracy_value = accuracy2(img, labels)
# accuracy_value = tf.reduce_mean(tf.cast(accuracy_value2, tf.float32))

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

	loopAmount = 2000

	# Correcte code
	accuracyArray = [0, 0, 0]
	for x in range(loopAmount):
		training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
		sess.run(train_step, feed_dict={img: training_set_image, labels: training_set_class})
		training_set_image /= 255
		if x % 100 == 0:
			print(str(x) + ": ")
			print(training_set_name[1])
			evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filename = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			accuracy = sess.run(accuracy_value, feed_dict={img: evaluation_set_image, labels: evaluation_set_class})
			evaluation_set_image /= 255
			accuracyArray.append(accuracy)
			print((accuracy))
		if x % 1000 == 0:
			print('%s%s ======= Iteration %s of %s | %s done ======= %s' % (fg('white'), bg('red'), str(x), str(loopAmount), str("{0:.0f}%".format((x/loopAmount) * 100)), attr('reset')))

	# plt.plot(accuracyArray)
	# plt.title('Juiste voorspellingen')
	# plt.ylabel('Learning rate: ' + str(learningRate) + ', loops: ' + str(loopAmount) + ", Dropout KeepProb rate: " + str(keepProb1))
	# print(accuracyArray)
	input("Press Enter to continue...")

	evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filenames = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN

	# test_accuracy = sess.run(fR.accuracy, feed_dict={
	# 	fR.x: evaluation_set_image,
	# 	fR.y_: evaluation_set_class,
	# 	fR.keep_prob1: 1.0,
	# 	fR.keep_prob1: 2.0
	# })
	# print('Test accuracy {:g}'.format(test_accuracy))

	coord.request_stop()
	coord.join(threads)
