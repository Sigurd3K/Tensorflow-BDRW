#  STAGES

"""Stages:
List of filenames
File name shuffling (optional)
Epoch limit (optional)
FIlename queue
File format reader
A decoder for a record format read by the reader
Preprocessing (optional)
Example queue
"""

# https://gist.github.com/eerwitt/518b0c9564e500b4b50f

import tensorflow as tf

print(" ")
print("== fileReader.py ==")

FILEDIR = './data/BDRW_train'
TRAINING_DIR= FILEDIR + '/BDRW_train_1/'
VALIDATION_DIR = FILEDIR + '/BDRW_train_2/'
LABEL_FILE = FILEDIR + '/BDRW_train_2/labels.csv'
EPOCH_LIMIT = 50
FILES_VALIDATION = 0
BATCH_SIZE = 50
NUM_PREPROCESS_THREADS = 1
MIN_QUEUE_EXAMPLES= 256

print(LABEL_FILE)



# W = tf.Variable(tf.zeros([2304,10]))
# b = tf.Variable(tf.zeros([10]))

image_name = tf.placeholder(tf.string, name='image_name')
image_class = tf.placeholder(tf.string, name='image_class')

# print(LABEL_FILE)


def filenameLister():
	FILES_TRAINING = tf.train.string_input_producer(
		tf.train.match_filenames_once(TRAINING_DIR + "digit_*.jpg"))
	print("Filedir: %s" % (FILEDIR))
	return FILES_TRAINING


def labelFileInit(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=0)
	_, csv_row = reader.read(filename_queue)
	record_defaults = [["Image1"], ["5"]]
	image_name, image_class = tf.decode_csv(csv_row, record_defaults=record_defaults)
	return image_name, image_class

FILES_TRAINING = filenameLister()

# labelFile_queue = eval("[\"" + LABEL_FILE + "\"]")
print("[\"" + LABEL_FILE + "\"]")
# labelFile_queue = tf.train.string_input_producer(["olympics2016.csv"], num_epochs=1, shuffle=False) // werkt niet met num_epochs=1 erbij. OM SHUFFLE TE KUNNEN GEBRUIKEN MOET JE INIT VAR EN RUN DOEN IN VARS
# labelFile_queue = tf.train.string_input_producer(["./data/BDRW_train/BDRW_train_2/labels.csv"], shuffle=False)

labelFile_queue = tf.train.string_input_producer(["./data/BDRW_train/BDRW_train_2/labels.csv"], num_epochs=1, shuffle=False)

image_name, image_class = labelFileInit(labelFile_queue)
print(labelFile_queue)

# print(type(FILES_TRAINING))


image_reader = tf.WholeFileReader()


_, image_file = image_reader.read(FILES_TRAINING)

image_orig = tf.image.decode_jpeg(image_file)
image = tf.image.resize_images(image_orig, [48, 48])
image.set_shape((48, 48, 3))
num_preprocess_threads = 1
min_queue_examples = 256

images = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE, num_threads=NUM_PREPROCESS_THREADS, capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE, min_after_dequeue=MIN_QUEUE_EXAMPLES)


with tf.Session() as sess:
	tf.global_variables_initializer().run()
	tf.initialize_local_variables().run()

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	looper = 0

	while True:
		try:
			looper += 1
			# print(looper)
			# print("WHILE TRUE")
			image_name_ts, image_class_ts = sess.run([image_name, image_class])
			# print("END WHILE")
			print(image_name_ts, image_class_ts)
			# print("END WHILE 2")
		except tf.errors.OutOfRangeError:
			# Bij epoch = 1 in de queue geeft TextLineReader of de queue een outOfRange exception als er geen lines meer over zijn om te readen
			print(looper)
			print("Out of range error")
			print("@@@@@@@@@@@@@@@@################@@@@@=================++++++++++++++++=")
			break

	image_tensor = sess.run([images])


	print(image_tensor)
	print(len(image_tensor[0]))

	coord.request_stop()
	coord.join(threads)
