import tensorflow as tf
import math
from colored import fg, bg, attr
import fileReader as fR

# -- Classifier variables --

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

	try:
		while not coord.should_stop():
			looper += 1
			print(looper)
			# print("WHILE TRUE")
			image_name_batch_b, image_class_batch_b = sess.run([fR.image_name_batch, fR.image_class_batch]) # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			# print(type(image_name_batch))
			print(" ")
			print("======== Batches uit de CSV ========")
			print(" ")
			print(image_name_batch_b, image_class_batch_b)
	except tf.errors.OutOfRangeError:
		# Bij epoch = 1 in de queue geeft TextLineReader of de queue een outOfRange exception als er geen lines meer over zijn om te readen
		print(looper)
		print('%s%s =========== Out of range error =========== %s' % (fg('white'), bg('yellow'), attr('reset')))
		print('%s%s === Stopped loading of one-hot labels ==== %s' % (fg('white'), bg('yellow'), attr('reset')))
	finally:
		coord.request_stop()


	image_tensor = sess.run([fR.images])


	print(image_tensor)


	coord.request_stop()
	coord.join(threads)