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
import numpy as np

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

W = tf.Variable(tf.zeros([2304,10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, shape=[None, 2304])
y = tf.matmul(x,W) + b
y_ = tf.placeholder(tf.float32, shape=[None, 10])

image_name = tf.placeholder(tf.string, name='image_name')
image_class = tf.placeholder(tf.string, name='image_class')

# image_name_batch = tf.placeholder(dtype=tf.string, shape=[None, ], name='image_name_batch')
# image_class_batch = tf.placeholder(dtype=tf.int8, shape=[None, ], name='image_class_batch')

# evaluation_labels = tf.placeholder(tf.float)


# print(LABEL_FILE)


def filenameLister():
	FILES_TRAINING = tf.train.string_input_producer(
		tf.train.match_filenames_once(TRAINING_DIR + "digit_*.jpg"))
	print("Filedir: %s" % (FILEDIR))
	return FILES_TRAINING


def labelFileInit(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=0)
	_, csv_row = reader.read(filename_queue)
	record_defaults = [['Image1'], [5]]
	image_name, image_class = tf.decode_csv(csv_row, record_defaults=record_defaults
	)

	image_class = tf.one_hot(image_class, 10, on_value=1, off_value=0)
	print(image_class)
	return image_name, image_class

def labelFileBatchProcessor(batch_size, num_epochs=None):
	labelFile_queue = tf.train.string_input_producer(["./data/BDRW_train/BDRW_train_2/labels.csv"], num_epochs=1, shuffle=False)
	image_name, image_class = labelFileInit(labelFile_queue)
	# print(labelFile_queue)
	min_after_dequeue = 50
	capacity = min_after_dequeue + 3 * batch_size
	image_name_batch, image_class_batch = tf.train.shuffle_batch(
		[image_name, image_class], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)

	print(" END OF FUNCTION LFBP")

	return image_name_batch, image_class_batch


image_name_batch, image_class_batch = labelFileBatchProcessor(50, 1)

FILES_TRAINING = filenameLister()

# labelFile_queue = eval("[\"" + LABEL_FILE + "\"]")
print("[\"" + LABEL_FILE + "\"]")
# labelFile_queue = tf.train.string_input_producer(["olympics2016.csv"], num_epochs=1, shuffle=False) // werkt niet met num_epochs=1 erbij. OM SHUFFLE TE KUNNEN GEBRUIKEN MOET JE INIT VAR EN RUN DOEN IN VARS
# labelFile_queue = tf.train.string_input_producer(["./data/BDRW_train/BDRW_train_2/labels.csv"], shuffle=False)


# print(type(FILES_TRAINING))


image_reader = tf.WholeFileReader()


_, image_file = image_reader.read(FILES_TRAINING)

image_orig = tf.image.decode_jpeg(image_file)
image = tf.image.resize_images(image_orig, [48, 48])
image.set_shape((48, 48, 3))
num_preprocess_threads = 1
min_queue_examples = 256

images = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE, num_threads=NUM_PREPROCESS_THREADS, capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE, min_after_dequeue=MIN_QUEUE_EXAMPLES)


######### START MACHINE LEARNING ############

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)





with tf.Session() as sess:
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	looper = 0


	try:
		while not coord.should_stop():
			looper += 1
			print(looper)
			# print("WHILE TRUE")
			image_name_batch_b, image_class_batch_b = sess.run([image_name_batch, image_class_batch]) # EERSTE VARS NIET HETZELFDE NOEMEN ALS DIE IN RUN
			print(type(image_name_batch))
			print(" ")
			print("======== Batches uit de CSV ========")
			print(" ")
			print(image_name_batch_b, image_class_batch_b)
	except tf.errors.OutOfRangeError:
		# Bij epoch = 1 in de queue geeft TextLineReader of de queue een outOfRange exception als er geen lines meer over zijn om te readen
		print(looper)
		print("Out of range error")
		print("@@@@@@@@@@@@@@@@################@@@@@=================++++++++++++++++=")
	finally:
		coord.request_stop()


	image_tensor = sess.run([images])


	print(image_tensor)
	print(len(image_tensor[0]))

	coord.request_stop()
	coord.join(threads)
