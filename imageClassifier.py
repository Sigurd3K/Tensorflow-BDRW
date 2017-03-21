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

	# for _ in range(1000):
	# 	batch_xs, batch_ys = sess.run([images])
	# 	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	printedTest = False

	try:
		while not coord.should_stop():
			looper += 1
			print(looper)
			training_set_name, training_set_class, training_set_image, filename = sess.run([fR.training_set_name, fR.training_set_class, fR.training_set_image, fR.filenames]) # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			# print(training_set_image)
			# print(type(image_name_batch))
			# print(" ")
			# print("======== Batches uit de CSV ========")
			# print(" ")
			# print(image_name_batch_b, image_class_batch_b)
			# print(image_class_batch_c)

			if printedTest == False:
				plt.style.use("ggplot")
				print('%s%s =========== TEST SAMPLE =========== %s' % (fg('white'), bg('green'), attr('reset')))
				print(training_set_name)
				print(" ")
				print(training_set_class)
				imt = plt.imshow(training_set_image[2]/255)


				# print(img)
				print(" Afbeelding uit mijn batch: ")
				# print(training_set_image[0]/255)
				print(filename[2])
				filename

				# print(image_batch_c[0][0].shape)
				# print(img.shape)
				# raw_input()
				input("Press Enter to continue...")
				# print(img)

				print('%s%s ======= END OF TEST SAMPLE ======= %s' % (fg('white'), bg('green'), attr('reset')))
				printedTest = True

	except tf.errors.OutOfRangeError:
		# Bij epoch = 1 in de queue geeft TextLineReader of de queue een outOfRange exception als er geen lines meer over zijn om te readen
		print(looper)
		print('%s%s =========== Out of range error =========== %s' % (fg('white'), bg('yellow'), attr('reset')))
		print('%s%s === Stopped loading of one-hot labels ==== %s' % (fg('white'), bg('yellow'), attr('reset')))
	finally:
		coord.request_stop()


	# image_tensor = sess.run([fR.images])


	# print(image_tensor)

	# x = fR.FILES_TRAINING.dequeue_many(10)
	# test = x + " Tes"

	# print('%s%s Print test: %s %s' % (fg('white'), bg('red'), test, attr('reset')))
	# tf.Print(data = [test], message=None, first_n=None, summarize=None, name=None)
	# printSmallQueue = tf.Print(test, [test], message="Looo: ", summarize=10)
	# sess.run(printSmallQueue)
	# tf.Print('%s%s Length of image tensor: %s %s' % (fg('white'), bg('red'), len(image_tensor[0]), attr('reset')))



	coord.request_stop()
	coord.join(threads)