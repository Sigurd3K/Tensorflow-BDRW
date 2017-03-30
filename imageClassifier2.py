import tensorflow as tf
import math
from colored import fg, bg, attr
import fileReader as fR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -- Classifier variables --
'''
layer_sizes = [2304, 3000, 500, 250, 250, 250, 10]
shapes = zip(layer_sizes[:-1], layer_sizes[1:]) #Element 1 en 2 worden samen in tuple gestoken enz omdat lists ten opzichte van elkaar met 1 verschuiven.
number_of_layers = len(layer_sizes) - 1

starter_learning_rate = 0.02

batch_size = 50

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)


def bi(inits, size, name):
	return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
	return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

weights = {
	'W': [wi(s, "W") for s in shapes],
	'V': [wi(s[::-1], "V") for s in shapes],
	'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(number_of_layers)],
	'gamma': [bi(0.0, layer_sizes[l+1], "beta") for l in range(number_of_layers)],
	}
'''

# -- End of Classifier variables --z`

# cross_entropy = tf.reduce_mean(
# 	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
# 	)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	looper = 0

	printedTest = False

	loopAmount = 1000
	learningRate = 0.05

	# Correcte code
	accuracyArray = [0, 0, 0]
	for x in range(loopAmount):
		training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
		sess.run(fR.train_step, feed_dict={fR.x: training_set_image, fR.y_: training_set_class, fR.learningRate: learningRate})
		if x % 2 == 0:
			print(str(x) + ": ")
			print(training_set_name[1])
			evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filename = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			accuracy = sess.run(fR.accuracy, feed_dict={fR.x: training_set_image, fR.y_: training_set_class})
			accuracyArray.append(accuracy)
			print((accuracy))
			# print(sess.run(fR.accuracy, feed_dict={fR.x: evaluation_set_image, fR.y_: evaluation_set_class}))
		if x % 1000 == 0:
			print('%s%s ======= Iteration %s of %s | %s done ======= %s' % (fg('white'), bg('green'), str(x), str(loopAmount), str("{0:.0f}%".format((x/loopAmount) * 100)), attr('reset')))

	plt.plot(accuracyArray)
	plt.ylabel('Juiste voorspellingen, Learing rate: ' + str(learningRate) + ', loops: ' + str(loopAmount) + "Dropout KeepProb rate: " + str(keepProb))
	# print(accuracyArray)
	input("Press Enter to continue...")

	evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filenames = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN

	test_accuracy = sess.run(fR.accuracy, feed_dict={
		fR.x: evaluation_set_image,
		fR.y_: evaluation_set_class})
	print('Test accuracy {:g}'.format(test_accuracy))

	# Test trained model
	# evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filename = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
	#
	# print(sess.run(fR.accuracy, feed_dict={fR.x: evaluation_set_image, fR.y_: evaluation_set_class}))

	coord.request_stop()
	coord.join(threads)