# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import math
import numpy as np
from featureExtractors import FEATURE_NAMES

PRINT = True


class PerceptronPacman:

    def __init__(self, num_train_iterations=20, learning_rate=1):

        self.max_iterations = num_train_iterations
        self.learning_rate = learning_rate

        # A list of which features to include by name. To exclude a feature comment out the line with that feature name
        feature_names_to_use = [
            'closestFood', 
            'closestFoodNow',
            'closestGhost',
            'closestGhostNow',
            'closestScaredGhost',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            "foodCount",
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',  
            'furthestFood', 
            'numberAvailableActions',
            "ratioCapsuleDistance",
            "ratioFoodDistance",
            "ratioGhostDistance",
            "ratioScaredGhostDistance"
            ]
        
        # we start our indexing from 1 because the bias term is at index 0 in the data set
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(1, len(FEATURE_NAMES) + 1)))

        # a list of the indices for the features that should be used. We always include 0 for the bias term.
        self.features_to_use = [0] + [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        "*** YOUR CODE HERE ***"
        self.weights = np.random.randn(len(self.features_to_use)) / np.sqrt(len(self.features_to_use))


    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your perceptron and output activation function

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.
        """
        # filter the data to only include your chosen features. We might not need to do this if we're working with training data that has already been filtered.
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        z = np.dot(self.weights, vector_to_classify)
        return self.activationOutput(z)


    def activationHidden(self, x):
        """
        Implement your chosen activation function for any hidden layers here.
        """

        "*** YOUR CODE HERE ***"
        # pass
        # return np.maximum(0, x)
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def activationOutput(self, x):
        """
        Implement your chosen activation function for the output here.
        """

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # return 1 / (1 + np.exp(-x))
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here. 
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        X_eval = data[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        predictions = np.array([self.predict(x) for x in X_eval])
        
        accuracy = np.mean((predictions >= 0.5) == labels)
        
        epsilon = 1e-15
        loss = -np.mean(labels * np.log(predictions + epsilon) + (1 - labels) * np.log(1 - predictions + epsilon))
        
        return accuracy, loss

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        X_train = trainingData[:, self.features_to_use]
        X_validate = validationData[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        for iteration in range(self.max_iterations):
            for x, y in zip(X_train, trainingLabels):
                prediction = self.predict(x)
                error = y - prediction
                gradient = error * x
                self.weights += self.learning_rate * gradient

            train_accuracy, train_loss = self.evaluate(trainingData, trainingLabels)
            val_accuracy, val_loss = self.evaluate(validationData, validationLabels)

            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(f"iteration {iteration+1}/{self.max_iterations}")
            print(f"training acc: {train_accuracy:.4f}, loss: {train_loss:.4f}")
            print(f"validation acc: {val_accuracy:.4f}, loss: {val_loss:.4f}")
            print("--------------------")

        import matplotlib.pyplot as plt

        epochs = range(1, self.max_iterations + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_weights(self, weights_path):
        """
        Saves your weights to a .model file. You're free to format this however you like.
        For example with a single layer perceptron you could just save a single line with all the weights.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        np.savetxt(weights_path, self.weights)

    def load_weights(self, weights_path):
        """
        Loads your weights from a .model file. 
        Whatever you do here should work with the formatting of your save_weights function.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        self.weights = np.loadtxt(weights_path)
