"""Testing of the image loader for the BDRW Dataset"""

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

	try:
		while not coord.should_stop():
			looper += 1
			print(looper)
			training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames]) # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			sess.run(fR.train_step, feed_dict={fR.x: training_set_image, fR.y_: training_set_class})
			# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

			if printedTest == False:
				plt.style.use("ggplot")
				print('%s%s =========== TEST SAMPLE =========== %s' % (fg('white'), bg('green'), attr('reset')))
				print(training_set_name)
				print(" ")
				print(training_set_class)
				# imt = plt.imshow(training_set_image[2]/255)

				print(" Afbeelding uit mijn batch: ")
				print(filename[2])
				filename

				input("Press Enter to continue...")

				print('%s%s ======= END OF TEST SAMPLE ======= %s' % (fg('white'), bg('green'), attr('reset')))
				printedTest = True

	except tf.errors.OutOfRangeError:
		# Bij epoch = 1 in de queue geeft TextLineReader of de queue een outOfRange exception als er geen lines meer over zijn om te readen
		print(looper)
		print('%s%s =========== Out of range error =========== %s' % (fg('white'), bg('yellow'), attr('reset')))
		print('%s%s === Stopped loading of one-hot labels ==== %s' % (fg('white'), bg('yellow'), attr('reset')))
	finally:
		coord.request_stop()

	coord.request_stop()
	coord.join(threads)