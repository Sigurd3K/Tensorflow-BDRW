import tensorflow as tf

filename = "olympics2016.csv"

features = tf.placeholder(tf.int32, shape=[3], name='features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')


def create_file_reader_ops(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=1)
	_, csv_row = reader.read(filename_queue)
	record_defaults = [[""], [""], [0], [0], [0], [0]]
	country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
	features = tf.stack([gold, silver, bronze])
	return features, country


filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename), num_epochs=1, shuffle=False)
example, country = create_file_reader_ops(filename_queue)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	looper = 0

	while True:
		try:
			looper += 1
			print(looper)
			print(" aaaaaaaaaaaaa aaaaaaaa aaaaaaa aaaaa aa a")
			example_data, country_name = sess.run([example, country])
			print(" kjdsrghjkdhgk hjklghdfkjg hdfkgh kjdfh kgj")
			tf.print(example_data, country_name)
		except tf.errors.OutOfRangeError:
			print(looper)
			print("out of range")
			break
