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

FILES_TRAINING = filenameLister()

labelFileInit()

print(type(FILES_TRAINING))


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

	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	image_tensor = sess.run([images])
	print(image_tensor)
	print(len(image_tensor[0]))

	coord.request_stop()
	coord.join(threads)

