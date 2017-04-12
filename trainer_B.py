"""Full Keras code with Tensorflow FileReader backend, does not work yet"""


import fileReader as fR

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

x_train, y_train, _, _ = fR.return_training_set()
x_test, y_test, _, _ = fR.return_eval_set()

# y_train = keras.utils.to_categorical(y_train, 10)

model.add(Dense(50, activation='relu', input_dim=6912))
model.add(Activation('relu'))
# model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, batch_size=50)
