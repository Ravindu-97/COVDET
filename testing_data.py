import numpy as np
import os
import cv2
import random
import pickle

# Setting the required constants
TESTING_IMAGE_DIRECTORY = "C:/FYP/dataset/test/"
LABELS = ["COVID-19", "Viral Pneumonia", "Normal"]

# Setting the testing image size as 200
IMAGE_SIZE = 200
# Creating an array to hold the testing data
testing_data = []


# Defining a function to prepare the testing images/data
def prepare_testing_data():
    for label in LABELS:
        path = os.path.join(TESTING_IMAGE_DIRECTORY + label)  # Going inside each label/category
        label_number = LABELS.index(label)  # Getting the index number in 'LABELS' array of each label

        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image))  # Reading each image in each label/category
                new_image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))  # Resizing each image in each label/category
                testing_data.append([new_image_array, label_number])  # Appending each image in each label/category to 'testing_data' array
            except Exception as e:
                pass


prepare_testing_data()
print(len(testing_data))
random.shuffle(testing_data)  # Shuffling the testing images/data

X_test = []  # feature sets of testing images
y_test = []  # label sets of testing images

# Appending the feature set and label set of testing data into separate arrays
for features, labels in testing_data:
    X_test.append(features)
    y_test.append(labels)

# Converting the normal X_test and y_test arrays to numpy arrays to feed into a model
X_test = np.array(X_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y_test = np.array(y_test)

print(X_test.shape)

# Saving the testing sets as pickle files for later usage
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
