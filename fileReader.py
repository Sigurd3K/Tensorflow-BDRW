import tensorflow as tf
import numpy as np
from colored import fg, bg, attr # For coloring Python print() output



# File locations:
FILEDIR = './data/BDRW_train'
TRAINING_DIR= FILEDIR + '/BDRW_train_1/'
VALIDATION_DIR = FILEDIR + '/BDRW_train_2/'
LABEL_FILE = FILEDIR + '/BDRW_train_2/labels.csv'

EPOCH_LIMIT = 50
FILES_VALIDATION = 0
BATCH_SIZE = 10
NUM_PREPROCESS_THREADS = 1
MIN_QUEUE_EXAMPLES= 256

print(LABEL_FILE)


# Gewichten en biasen een random beginvariabele geven. Zo kan je oa. dode neuronen voorkomen en sneller te leren.
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.5)
	return tf.Variable(initial, name="Weight") # Deze placeholders krijgen een naam, handig voor in de Graph

def bias_variable(shape):
  initial = tf.constant(0.5, shape=shape)
  return tf.Variable(initial, name="Biases")

# De placeholders aanmaken door bovenstaande functies uit te voeren
W = weight_variable([6912, 10])
b = bias_variable([10])

learningRate = tf.placeholder(tf.float32, name="LearningRate")


x = tf.placeholder(tf.float32, shape=[None, 6912], name="Image")

# In sommige gevallen wil je geen "platte afbeelding", De convolutional layers van Keras hebben bvb 2d input nodig:
# x = tf.placeholder(tf.float32, shape=[None, 48, 48, 3], name="Image")

y = tf.matmul(x, W) + b # Logits. (Gebruiken in de layers een andere methode)
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="CorrectClass") # De correcte klasses, als one_hot

image_name = tf.placeholder(tf.string, name='image_name')
image_class = tf.placeholder(tf.string, name='image_class')


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

	return image_name, image_class, filename


def labelFileBatchProcessor(batch_size, num_epochs=2, what_set="validation"):
	if what_set == "training":
		inputCsv = ["./data/BDRW_train/BDRW_train_1/labels.csv"]
	elif what_set == "validation":
		inputCsv = ["./data/BDRW_train/BDRW_train_2/labels.csv"]
	labelFile_queue = tf.train.string_input_producer(inputCsv, shuffle=False)

	image_name, image_class, filename = labelFileInit(labelFile_queue,  what_set=what_set)
	min_after_dequeue = 50
	capacity = min_after_dequeue + 3 * batch_size

	image = build_images(filename)

	image_name_batch, image_class_batch, images, filename = tf.train.shuffle_batch(
		[image_name, image_class, image, filename], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)

	return image_name_batch, image_class_batch, images, filename


print("[\"" + LABEL_FILE + "\"]")


def build_images(files_training):
	rotation = tf.random_normal([1], mean=0.0, stddev=0.4, dtype=tf.float32, seed=None, name=None)

	image_file = tf.read_file(files_training[0])
	image_orig = tf.image.decode_jpeg(image_file)
	image = tf.image.resize_images(image_orig, [48, 48])
	image = tf.contrib.image.rotate(image, rotation)
	image.set_shape((48, 48, 3))
	image = tf.reshape(image, [-1])
	num_preprocess_threads = 1
	min_queue_examples = 256
	return image


def return_training_set():
	image_tra_name_batch, image_tra_class_batch, images, imagepath = labelFileBatchProcessor(50, 1, "training")

	return image_tra_name_batch, image_tra_class_batch, images, imagepath

training_set_name, training_set_class, training_set_image, filenames = return_training_set()


def return_eval_set():
	image_tra_name_batch, image_tra_class_batch, images, imagepath = labelFileBatchProcessor(100, 1, "validation")

	return image_tra_name_batch, image_tra_class_batch, images, imagepath

evaluation_set_name, evaluation_set_class, evaluation_set_image, evaluation_filenames = return_eval_set()

# =============================================

# Convolutions ding
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convolutions
image = tf.reshape(x, [-1, 48, 48, 3])

# Dropout Layer 1:
keep_prob1 = tf.placeholder(tf.float32)
image_drop1 = tf.nn.dropout(image, keep_prob1)

# First Convolutional Layer:
W_conv1 = weight_variable([8, 8, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(image_drop1, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # 24x24

# Second Convolutional Layer:
W_conv2 = weight_variable([8, 8, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #12x12

# Densely Connected Layer:
W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Layer 3:
keep_prob2 = tf.placeholder(tf.float32)
h_fc1_drop2 = tf.nn.dropout(h_fc1, keep_prob2)

# Readout Layer:
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop2, W_fc2) + b_fc2

# =================================================

# TRAINING STEPS

# Loss
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)

# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

# Evaluation Steps
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
