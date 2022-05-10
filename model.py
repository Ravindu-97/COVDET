from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, date
import pytz

# Setting the number of epochs, batch size and optimizer for the CNN model
EPOCHS = 25
BATCH_SIZE = 32
OPTIMIZER = 'rmsprop'

# Loading the previously prepared training images/data
X = pickle.load(open("X_train.pickle", "rb"))
y = pickle.load(open("y_train.pickle", "rb"))

# Normalizing/Scaling the features of training images/data
X = X/255.0

# Initializing the CNN model as a sequential model
classifier = Sequential()

# Adding the Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, and Dense layers with different hyperparameters
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', input_shape=(200, 200, 3)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(2048, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Dense(1024, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Dense(512, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Dense(3, activation='softmax'))

# Compiling the CNN model with a loss function and an optimizer and providing the accuracy as a metric to observe the model's training process
classifier.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
classifier.summary()

# Creating the current time for model saving purposes
today = date.today()
tz_Colombo = pytz.timezone('Asia/Colombo') 
datetime = datetime.now(tz_Colombo)
time = datetime.strftime("%I:%M %p")

# Reducing the learning rate of the model by a factor of 0.5 if the validation accuracy doesn't increase in 2 epochs in a row while training
reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2)

# Starting the training process of the model
training_process = classifier.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, callbacks=[reduce_learning_rate])

# Plotting the training and validation accuracies for visualization
accuracy = training_process.history['accuracy']
validation_accuracy = training_process.history['val_accuracy']
Epochs_number = range(1, EPOCHS + 1)
plt.plot(Epochs_number, accuracy, 'g', label='Training Accuracy')
plt.plot(Epochs_number, validation_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss for visualization
loss = training_process.history['loss']
validation_loss = training_process.history['val_loss']
Epochs_number = range(1, EPOCHS + 1)
plt.plot(Epochs_number, loss, 'g', label='Training Loss')
plt.plot(Epochs_number, validation_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Getting and printing the validation accuracy of the final epoch
final_validation_accuracy = validation_accuracy[EPOCHS - 1]
model_validation_accuracy = "{:.2%}".format(final_validation_accuracy)
print(model_validation_accuracy)

# saving the model
classifier.save("model/Model - {}-{}-{}-{}.h5".format(BATCH_SIZE, EPOCHS, OPTIMIZER, model_validation_accuracy))
