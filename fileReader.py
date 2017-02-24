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

print(" ")
print("== fileReader.py ==")

FILEDIR = 'data/BDRW_train'
TRAINING_DIR= FILEDIR + '\BDRW_train_1'
VALIDATION_DIR= FILEDIR + '\BDRW_train_2'
LABEL_FILE = 'filedir'

def filenameLister():
	print("Filedir: %s" % (FILEDIR))


filenameLister()
