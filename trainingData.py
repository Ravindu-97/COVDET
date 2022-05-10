import numpy as np
import os
import cv2
import random
import pickle

# Setting the required constants
TRAINING_IMAGE_DIRECTORY = "C:/FYP/dataset/train/"
LABELS = ["COVID-19", "Viral Pneumonia", "Normal"]

# Setting the training image size as 200
IMAGE_SIZE = 200
# Creating an array to hold the training data
training_data = []


# Defining a function to prepare the training images/data
def prepare_training_data():
    for label in LABELS:
        path = os.path.join(TRAINING_IMAGE_DIRECTORY + label)  # Going inside each label/category
        label_number = LABELS.index(label)  # Getting the index number in 'LABELS' array of each label

        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image))  # Reading each image in each label/category
                new_image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))  # Resizing each image in each label/category
                training_data.append([new_image_array, label_number])  # Appending each image in each label/category to 'testing_data' array
            except Exception as e:
                pass


prepare_training_data()
print(len(training_data))
random.shuffle(training_data)  # # Shuffling the training images/data

X = []  # feature sets of training images
y = []  # label sets of training images

# Appending the feature set and label set of training data into separate arrays
for features, labels in training_data:
    X.append(features)
    y.append(labels)

# Converting the normal X and y arrays to numpy arrays to feed into a model
X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y = np.array(y)

print(X.shape)

# Saving the training sets as pickle files for later usage
pickle_out = open("X_train.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
