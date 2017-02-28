FILEDIR = './data/BDRW_train'
TRAINING_DIR= FILEDIR + '/BDRW_train_1/'
VALIDATION_DIR = FILEDIR + '/BDRW_train_2/'
LABEL_FILE = FILEDIR + '/BDRW_train_2/labels.csv'



import tensorflow as tf

country = tf.placeholder(tf.string, name='country')
code = tf.placeholder(tf.string, name='code')


def create_file_reader_ops(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=1)
	_, csv_row = reader.read(filename_queue)
	record_defaults = [[""], [""]]
	country, code = tf.decode_csv(csv_row, record_defaults=record_defaults)
	print("create_file_reader_ops")
	return country, code

labelFile_queue = tf.train.string_input_producer(tf.train.match_filenames_once(LABEL_FILE), num_epochs=1, shuffle=False)
country, code = create_file_reader_ops(labelFile_queue)


with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	looper = 0
	while True:
		try:
			looper += 1
			print(looper)
			country, code = sess.run([country, code])
			print(typeof(example_data))
			looper += 1
			print(looper)
			print(country)
			print(" ")
			print(" ")
			print(" test")
		except tf.errors.OutOfRangeError:
			print(looper)
			print("Out of range error")
			break

	print("After break")
	# print(example_data, country_name)