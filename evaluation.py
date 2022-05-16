from keras.models import load_model
import numpy as np
import pickle 
from sklearn.metrics import confusion_matrix, classification_report

# Loading the created CNN model
model = load_model('model/model.h5')

# Loading the previously created testing data
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

# Scaling the features
X_test = X_test/255.0  

# Predicting all the testing images
y_prediction = model.predict(X_test)
# Getting the label numbers of all the predicted images
y_prediction_label_numbers = [np.argmax(label_number) for label_number in y_prediction]

# Generating classification report and confusion matrix
print("Classification Report: \n", classification_report(y_test, y_prediction_label_numbers))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_prediction_label_numbers))

# Calculating the overall testing accuracy of the model
print("Model Testing Accuracy: \n", model.evaluate(X_test, y_test))