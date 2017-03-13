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
from colored import fg, bg, attr


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

W = tf.Variable(tf.zeros([2304,10]), name="Weights")
b = tf.Variable(tf.zeros([10]), name="Biases")

x = tf.placeholder(tf.float32, shape=[None, 2304], name="flat_IMG")
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass")

image_name = tf.placeholder(tf.string, name='image_name')
image_class = tf.placeholder(tf.string, name='image_class')

# image_name_batch = tf.placeholder(dtype=tf.string, shape=[None, ], name='image_name_batch')
# image_class_batch = tf.placeholder(dtype=tf.int8, shape=[None, ], name='image_class_batch')

# evaluation_labels = tf.placeholder(tf.float)


# def filenameLister():
# 	FILES_TRAINING = tf.train.string_input_producer(
# 		tf.train.match_filenames_once(TRAINING_DIR + "digit_*.jpg"))
# 	print("Filedir: %s" % (FILEDIR))
# 	return FILES_TRAINING


def filenameLister2(imageNameBatch):
	print('%s%s filenameLister2 %s' % (fg('white'), bg('blue'), attr('reset')))
	FILES_TRAINING = TRAINING_DIR + imageNameBatch + ".jpg"
	FILES_TRAINING = tf.train.string_input_producer(FILES_TRAINING, name="CSVFilenames")
	print('%s%s filenameLister2 END %s' % (fg('white'), bg('blue'), attr('reset')))
	return FILES_TRAINING


def labelFileInit(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=0)
	_, csv_row = reader.read(filename_queue)
	record_defaults = [['Image1'], [5]]
	image_name, image_class = tf.decode_csv(csv_row, record_defaults=record_defaults)

	image_class = tf.one_hot(image_class, 10, on_value=1, off_value=0)
	print(image_class)
	return image_name, image_class


def labelFileBatchProcessor(batch_size, num_epochs=None, what_set="training"):
	if what_set == "training":
		inputCsv = ["./data/BDRW_train/BDRW_train_1/labels.csv"]
	elif what_set == "validation":
		inputCsv = ["./data/BDRW_train/BDRW_train_2/labels.csv"]
	# inputCsv = ["./data/BDRW_train/BDRW_train_2/labels.csv"]
	labelFile_queue = tf.train.string_input_producer(inputCsv, num_epochs=1, shuffle=False)
	image_name, image_class = labelFileInit(labelFile_queue)
	# print(labelFile_queue)
	min_after_dequeue = 50
	capacity = min_after_dequeue + 3 * batch_size
	image_name_batch, image_class_batch = tf.train.shuffle_batch(
		[image_name, image_class], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)

	print(" END OF FUNCTION LFBP")

	return image_name_batch, image_class_batch


image_tra_name_batch, image_tra_class_batch = labelFileBatchProcessor(50, 1, "training")
image_val_name_batch, image_val_class_batch = labelFileBatchProcessor(50, 1, "validation")


# FILES_TRAINING = filenameLister()
# FILES_VALIDATION = filenameLister()
FILES_TRAINING2 = filenameLister2(image_tra_name_batch)
FILES_VALIDATION2 = filenameLister2(image_val_name_batch)

# labelFile_queue = eval("[\"" + LABEL_FILE + "\"]")
print("[\"" + LABEL_FILE + "\"]")
# labelFile_queue = tf.train.string_input_producer(["olympics2016.csv"], num_epochs=1, shuffle=False) // werkt niet met num_epochs=1 erbij. OM SHUFFLE TE KUNNEN GEBRUIKEN MOET JE INIT VAR EN RUN DOEN IN VARS
# labelFile_queue = tf.train.string_input_producer(["./data/BDRW_train/BDRW_train_2/labels.csv"], shuffle=False)


image_reader = tf.WholeFileReader()


_, image_file = image_reader.read(FILES_TRAINING2)

image_orig = tf.image.decode_jpeg(image_file)
image = tf.image.resize_images(image_orig, [48, 48])
image.set_shape((48, 48, 3))
num_preprocess_threads = 1
min_queue_examples = 256

# Mag later weg, zorgt voor moeilijkheden:
# images = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE, num_threads=NUM_PREPROCESS_THREADS, capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE, min_after_dequeue=MIN_QUEUE_EXAMPLES)

images = tf.train.batch([image], batch_size=BATCH_SIZE, num_threads=NUM_PREPROCESS_THREADS, capacity=BATCH_SIZE)
