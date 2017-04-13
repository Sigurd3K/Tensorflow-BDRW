import tensorflow as tf
import math
from colored import fg, bg, attr
import fileReader as fR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	looper = 0

	printedTest = False

	loopAmount = 2000
	learningRate = 0.0005
		# Bij AdamOptimizer is een zeer kleine learning rate gebruikelijk
		# AdamOptimizer gaat zijn learningRate zelf aanpassen dus dit zelf doen is niet echt nodig
	keepProb1 = 0.8
	keepProb2 = 0.8

	# Correcte code
	accuracyArray = [0, 0, 0]
	for x in range(loopAmount):
		training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
		sess.run(fR.train_step, feed_dict={fR.x: training_set_image, fR.y_: training_set_class, fR.learningRate: learningRate, fR.keep_prob1: keepProb1, fR.keep_prob2: keepProb2})
		if x % 100 == 0:
			print(str(x) + ": ")
			print(training_set_name[1])
			# training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image,fR.filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			# accuracy = sess.run(fR.accuracy, feed_dict={fR.x: training_set_image, fR.y_: training_set_class, fR.learningRate: learningRate, fR.keep_prob1: keepProb1, fR.keep_prob2: keepProb2})
			evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filename = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			accuracy = sess.run(fR.accuracy, feed_dict={fR.x: evaluation_set_image, fR.y_: evaluation_set_class, fR.keep_prob1: 1.0, fR.keep_prob2: 1.0})
			accuracyArray.append(accuracy)
			print((accuracy))
			# print(sess.run(fR.accuracy, feed_dict={fR.x: evaluation_set_image, fR.y_: evaluation_set_class}))
		if x % 1000 == 0:
			print('%s%s ======= Iteration %s of %s | %s done ======= %s' % (fg('white'), bg('red'), str(x), str(loopAmount), str("{0:.0f}%".format((x/loopAmount) * 100)), attr('reset')))

	plt.plot(accuracyArray)
	plt.title('Juiste voorspellingen')
	plt.ylabel('Learning rate: ' + str(learningRate) + ', loops: ' + str(loopAmount) + ", Dropout KeepProb rate: " + str(keepProb1))
	# print(accuracyArray)
	input("Press Enter to continue...")

	evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filenames = sess.run([fR.evaluation_set_name, fR.evaluation_set_class, fR.evaluation_set_image, fR.evaluation_filenames])  # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN

	test_accuracy = sess.run(fR.accuracy, feed_dict={
		fR.x: evaluation_set_image,
		fR.y_: evaluation_set_class,
		fR.keep_prob1: 1.0,
		fR.keep_prob1: 2.0
	})
	print('Test accuracy {:g}'.format(test_accuracy))

	coord.request_stop()
	coord.join(threads)
