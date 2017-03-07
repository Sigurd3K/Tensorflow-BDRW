import tensorflow as tf
from colored import fg, bg, attr
import fileReader as fR

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
	print(len(image_tensor[0]))


	coord.request_stop()
	coord.join(threads)