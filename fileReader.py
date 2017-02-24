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

import tensorflow as tf

print(" ")
print("== fileReader.py ==")

FILEDIR = './data/BDRW_train'
TRAINING_DIR= FILEDIR + '/BDRW_train_1/'
VALIDATION_DIR = FILEDIR + '/BDRW_train_2/'
LABEL_FILE = 'filedir'
EPOCH_LIMIT = 50
FILES_VALIDATION = 0

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