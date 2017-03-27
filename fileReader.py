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

W = tf.Variable(tf.zeros([6912, 10]), name="Weights")
# W = tf.Variable(tf.zeros([2304, 10]), name="Weights")
b = tf.Variable(tf.zeros([10]), name="Biases")

x = tf.placeholder(tf.float32, shape=[None, 6912], name="Image")
# x = tf.placeholder(tf.float32, shape=[None, 48, 48, 3], name="Image")

y = tf.matmul(x, W) + b # Logits
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass")

image_name = tf.placeholder(tf.string, name='image_name')
image_class = tf.placeholder(tf.string, name='image_class')

# image_name_batch = tf.placeholder(dtype=tf.string, shape=[None, ], name='image_name_batch')
# image_class_batch = tf.placeholder(dtype=tf.int8, shape=[None, ], name='image_class_batch')

# evaluation_labels = tf.placeholder(tf.float)

def labelFileInit(filename_queue, what_set):
	reader = tf.TextLineReader(skip_header_lines=0)
	_, csv_row = reader.read(filename_queue)
	record_defaults = [['Image1'], [5]]
	image_name, image_class = tf.decode_csv(csv_row, record_defaults=record_defaults)

	image_class = tf.one_hot(image_class, 10, on_value=1, off_value=0)

	if what_set == "training":
		filename = [TRAINING_DIR + image_name + ".jpg"]
	elif what_set == "validation":
		filename = [VALIDATION_DIR + image_name + ".jpg"]

	print(image_class)
	return image_name, image_class, filename


def labelFileBatchProcessor(batch_size, num_epochs=None, what_set="validation"):
	if what_set == "training":
		inputCsv = ["./data/BDRW_train/BDRW_train_1/labels.csv"]
	elif what_set == "validation":
		inputCsv = ["./data/BDRW_train/BDRW_train_2/labels.csv"]
	labelFile_queue = tf.train.string_input_producer(inputCsv, shuffle=False)

	image_name, image_class, filename = labelFileInit(labelFile_queue,  what_set=what_set)
	# print(labelFile_queue)
	min_after_dequeue = 50
	capacity = min_after_dequeue + 3 * batch_size

	image = build_images(filename)

	image_name_batch, image_class_batch, images, filename = tf.train.shuffle_batch(
		[image_name, image_class, image, filename], batch_size=50, capacity=capacity,
		min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)

	print(" END OF FUNCTION LFBP")

	return image_name_batch, image_class_batch, images, filename


print("[\"" + LABEL_FILE + "\"]")


def build_images(files_training):
	image_file = tf.read_file(files_training[0])
	image_orig = tf.image.decode_jpeg(image_file)
	image = tf.image.resize_images(image_orig, [48, 48])
	image.set_shape((48, 48, 3))
	image = tf.reshape(image, [-1])
	num_preprocess_threads = 1
	min_queue_examples = 256
	return image


def return_training_set():
	image_tra_name_batch, image_tra_class_batch, images, imagepath = labelFileBatchProcessor(50, 1, "training")

	return image_tra_name_batch, image_tra_class_batch, images, imagepath

training_set_name, training_set_class, training_set_image, filenames = return_training_set()