# trainPerceptron.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import sys
import util
from pacman import Directions
from game import Directions, Actions
# from perceptronPacman import PerceptronPacman
import samples
import numpy as np
import math
import pandas as pd

from featureExtractors import FEATURE_NAMES
from featureExtractors import *

def default(str):
    return str + ' [Default: %default]'


TRAINING_SET_SIZE = 23000  # Just a bit bigger than the number of instances
TEST_SET_SIZE = 5000 # Just a bit bigger than the number of instances
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


USAGE_STRING = """
  USAGE:      python featureExtraction.py <options>
  EXAMPLES:  (1) python featureExtraction.py
                  - would extract the features from all training examples
                  using the furtherEnhancedPacmanFeatures function. 
                  This would be broken into train test and validation sets and the features
                  scaled based on the max and min values to be between 0 and 1.
             (2) python featureExtraction.py -t 1000
                  - would extract the features from the first 1000 training examples
                  using the furtherEnhancedPacmanFeatures function. 
                  This would be broken into train test and validation sets and the features
                  scaled based on the max and min values to be between 0 and 1.
                 """


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAINING_SET_SIZE, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Extracting Features")
    print("--------------------")

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.test <= 0:
        print("Testing set size should be a positive integer (you provided: %d)" % options.test)
        print(USAGE_STRING)
        sys.exit(2)

    args['training_size'] = options.training
    args['testing_size'] = options.test

    print(args)
    print(options)

    return args, options


# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl', 'pacmandata/food_validation.pkl', 'pacmandata/food_test.pkl'),
    'StopAgent': ('pacmandata/stop_training.pkl', 'pacmandata/stop_validation.pkl', 'pacmandata/stop_test.pkl'),
    'SuicideAgent': (
    'pacmandata/suicide_training.pkl', 'pacmandata/suicide_validation.pkl', 'pacmandata/suicide_test.pkl'),
    'GoodReflexAgent': (
    'pacmandata/good_reflex_training.pkl', 'pacmandata/good_reflex_validation.pkl', 'pacmandata/good_reflex_test.pkl'),
    'ContestAgent': (
    'pacmandata/contest_training.pkl', 'pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl')
}

def StringNameToNumber(numberString):

    if numberString == Directions.NORTH:
        return 0
    elif numberString == Directions.EAST:
        return 1
    elif numberString == Directions.SOUTH:
        return 2
    elif numberString == Directions.WEST:
        return 3
    elif numberString == Directions.STOP:
        return 4


def normalise_data(data, max_values, min_values):
    for feature_vector in data:
        for feature_name, value in feature_vector.items():
            feature_vector[feature_name] = (feature_vector[feature_name] - min_values[feature_name])/(max_values[feature_name] - min_values[feature_name])

    return data


def find_max_and_min_feature_values(data):
    feature_names = data[0].keys()
    max_values = dict(zip(feature_names, [-1*math.inf]*len(feature_names)))
    min_values = dict(zip(feature_names, [math.inf]*len(feature_names)))

    for feature_vector in data:
        for feature_name, value in feature_vector.items():
            if max_values[feature_name] < value:
                max_values[feature_name] = value
            if min_values[feature_name] > value:
                min_values[feature_name] = value

    return max_values, min_values


def convertToBinary(features, labels):
    binary_labels = []
    for i in range(len(features)):
        # print(labels[i])
        temp = [1 if StringNameToNumber(move) == labels[i] else 0 for move in features[i][1]]
        binary_labels.extend(temp)

    return binary_labels


def to_numpy_binary_data(trainingData):
    binary_features = []
    for i in range(len(trainingData)):
        legal_moves = trainingData[i][1]

        for move in legal_moves:
            features = trainingData[i][0][move]

            binary_features.append([features[feature_name] for feature_name in FEATURE_NAMES])
            # binary_features.append(features)

    return np.array(binary_features)


def runExtraction(args):
    featureFunction = enhancedFeatureExtractorPacman

    # data sizes and number of training iterations
    numTraining = args['training_size']
    numTest = args['testing_size']

    # load the data sets
    print("loading data...")
    trainingData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
    validationData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
    testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
    rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
    rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
    rawTestData, testLabels = samples.loadPacmanData(testData, numTest)

    # Extract features
    print("Extracting features...")

    trainingData = list(map(featureFunction, rawTrainingData))[:-1]
    validationData = list(map(featureFunction, rawValidationData))[:-1]
    testData = list(map(featureFunction, rawTestData))[:-1]

    trainingLabels = list(map(StringNameToNumber, trainingLabels[:-1]))
    validationLabels = list(map(StringNameToNumber, validationLabels[:-1]))
    testLabels = list(map(StringNameToNumber, testLabels[:-1]))

    # convert the data to binary labels format as numpy arrays
    trainingLabels = convertToBinary(trainingData, trainingLabels)
    trainingData = to_numpy_binary_data(trainingData)

    validationLabels = convertToBinary(validationData, validationLabels)
    validationData = to_numpy_binary_data(validationData)

    testLabels = convertToBinary(testData, testLabels)
    testData = to_numpy_binary_data(testData)

    # find the minimum and maximum values across all data sets for scaling
    max_train = np.max(trainingData, axis=0)
    max_validation = np.max(validationData, axis=0)
    max_test = np.max(testData, axis=0)
    max_values = np.max(np.vstack([max_train, max_validation, max_test]), axis=0)

    min_train = np.min(trainingData, axis=0)
    min_validation = np.min(validationData, axis=0)
    min_test = np.min(testData, axis=0)
    min_values = np.min(np.vstack([min_train, min_validation, min_test]), axis=0)

    # put a one at the front for bias max and 0 for bias min.
    max_values = np.concatenate([[1], np.max(trainingData, axis=0)])
    min_values = np.concatenate([[0], np.min(trainingData, axis=0)])

    # add bias to each data set
    trainingData = np.c_[np.ones(len(trainingData)), trainingData]
    validationData = np.c_[np.ones(len(validationData)), validationData]
    testData = np.c_[np.ones(len(testData)), testData]

    trainingData = (trainingData - min_values)/(max_values - min_values)
    validationData = (validationData - min_values)/(max_values - min_values)
    testData = (testData - min_values)/(max_values - min_values)

    df_train = pd.DataFrame(data=trainingData, columns=["bias"] + FEATURE_NAMES)
    df_train["label"] = trainingLabels
    df_train.to_csv("./pacmandata/q3_train.csv", index=False)

    df_validation = pd.DataFrame(data=validationData, columns=["bias"] + FEATURE_NAMES)
    df_validation["label"] = validationLabels
    df_validation.to_csv("./pacmandata/q3_validation.csv", index=False)

    df_test = pd.DataFrame(data=testData, columns=["bias"] + FEATURE_NAMES)
    df_test["label"] = testLabels
    df_test.to_csv("./pacmandata/q3_test.csv", index=False)

    np.savetxt("./pacmandata/q3_max_feature_values.txt", max_values)
    np.savetxt("./pacmandata/q3_min_feature_values.txt", min_values)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runExtraction(args)