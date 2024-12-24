import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.api.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.api.models import Sequential
from keras.api.utils import to_categorical

from utils.config import MODEL_PATH, DATA_ROOT, TRAIN_DATA_PATH, TEST_DATA_PATH, epochs


class Classifier:
    """
    A class to build, train, and evaluate a Convolutional Neural Network (CNN) model.

    """

    def __init__(self, num_classes=43):
        self.num_classes = num_classes
        self.model = None

    @staticmethod
    def date_time(x):
        """
        Returns formatted date and time based on input.
        Args:
            x: 1-4 (Returns formatted date and time)
        """
        if x == 1:
            return "Timestamp: {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
        if x == 2:
            return "Timestamp: {:%Y-%b-%d %H:%M:%S}".format(datetime.datetime.now())
        if x == 3:
            return "Date now: %s" % datetime.datetime.now()
        if x == 4:
            return "Date today: %s" % datetime.date.today()

    @staticmethod
    def plot_performance(history, figure_directory=None, ylim_pad=[0, 0]):
        """
        Plots the training and validation accuracy and loss.
        """
        print("Plotting model performance...")
        xlabel = "Epoch"
        legends = ["Training", "Validation"]

        plt.figure(figsize=(20, 5))

        y1 = history.history["accuracy"]
        y2 = history.history["val_accuracy"]

        min_y = min(min(y1), min(y2)) - ylim_pad[0]
        max_y = max(max(y1), max(y2)) + ylim_pad[0]

        plt.subplot(121)
        plt.plot(y1)
        plt.plot(y2)
        plt.title("Model Accuracy\n" + Classifier.date_time(1), fontsize=17)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.ylim(min_y, max_y)
        plt.legend(legends, loc="upper left")
        plt.grid()

        y1 = history.history["loss"]
        y2 = history.history["val_loss"]

        min_y = min(min(y1), min(y2)) - ylim_pad[1]
        max_y = max(max(y1), max(y2)) + ylim_pad[1]

        plt.subplot(122)
        plt.plot(y1)
        plt.plot(y2)
        plt.title("Model Loss\n" + Classifier.date_time(1), fontsize=17)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.ylim(min_y, max_y)
        plt.legend(legends, loc="upper left")
        plt.grid()

        if figure_directory:
            plt.savefig(figure_directory + "/history")
        plt.show()

    def load_data(self):
        """
        Loads and preprocesses the image data into numpy arrays.
        """
        print("Loading data...")
        data = []
        labels = []

        for i in range(self.num_classes):
            path = os.path.join(TRAIN_DATA_PATH, str(i))
            images = os.listdir(path)
            for a in images:
                try:
                    image = Image.open(os.path.join(path, a))
                    image = image.resize((30, 30))
                    image = np.array(image)
                    data.append(image)
                    labels.append(i)
                except:
                    print(f"Error loading image {a} in class {i}")

        data = np.array(data)
        labels = np.array(labels)

        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
        return data, labels

    def build_model(self, input_shape):
        """
        Builds and compiles the CNN model.
        """
        print("Building model...")
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.15))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.20))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Dense(self.num_classes, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.model.summary()

    def train_model(self, X_train, y_train, X_test, y_test, epochs):
        """
        Trains the model on the training data.
        """
        print("Training model...")
        with tf.device("/GPU:0"):
            history = self.model.fit(
                X_train,
                y_train,
                batch_size=128,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=1,
            )
        return history

    def evaluate_model(self, X_test, labels):
        """
        Evaluates the model on the test set.
        """
        print("Evaluating model...")
        pred = np.argmax(self.model.predict(X_test), axis=-1)
        accuracy = accuracy_score(labels, pred)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def load_test_data(self):
        """
        Loads the test data from a CSV file.
        """
        print("Loading test data...")
        y_test = pd.read_csv(TEST_DATA_PATH + ".csv")
        labels = y_test["ClassId"].values
        imgs = y_test["Path"].values

        data = []
        for img in imgs:
            image = Image.open(DATA_ROOT + img)
            image = image.resize((30, 30))
            data.append(np.array(image))

        X_test = np.array(data)
        print(f"Test Data shape: {X_test.shape}")
        return X_test, labels

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        self.model.save(MODEL_PATH)
        print("Model saved!")


def main():
    classifier = Classifier()
    start_time = time.time()

    X, y = classifier.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_train = to_categorical(y_train, classifier.num_classes)
    y_test = to_categorical(y_test, classifier.num_classes)

    classifier.build_model(X_train.shape[1:])

    history = classifier.train_model(X_train, y_train, X_test, y_test, epochs=epochs)

    Classifier.plot_performance(history)

    X_test, labels = classifier.load_test_data()
    classifier.evaluate_model(X_test, labels)

    classifier.save_model()
    print(f"Total training time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
