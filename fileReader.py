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
LABEL_FILE = 'filedir'
EPOCH_LIMIT = 50
FILES_VALIDATION = 0
BATCH_SIZE = 50
NUM_PREPROCESS_THREADS = 1
MIN_QUEUE_EXAMPLES= 256

def filenameLister():
	FILES_TRAINING = tf.train.string_input_producer(
	tf.train.match_filenames_once(TRAINING_DIR + "digit_*.jpg"))
	print("Filedir: %s" % (FILEDIR))
	#print("Training files dir: %s" % (TRAINING_DIR))
	#print(type(FILES_TlRAINING))
	return FILES_TRAINING

FILES_TRAINING = filenameLister()

# print("-------")
# print(tf.gfile.Glob(TRAINING_DIR + "digit_*.jpg"))

# print(FILES_TRAINING)

print(type(FILES_TRAINING))


image_reader = tf.WholeFileReader()

# print(FILES_TRAINING)

_, image_file = image_reader.read(FILES_TRAINING)

image = tf.image.decode_jpeg(image_file)
image.set_shape((102, 136, 3))
num_preprocess_threads = 1
min_queue_examples = 256


with tf.Session() as sess:
	tf.global_variables_initializer().run()

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	image_tensor = sess.run([image])
	print(image_tensor)

	coord.request_stop()
	coord.join(threads)
