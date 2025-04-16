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

import sys
import util
from pacman import Directions
from perceptronPacman import PerceptronPacman
import samples
import numpy as np
import math
import pandas as pd
from featureExtraction import TRAINING_SET_SIZE, TEST_SET_SIZE


def default(str):
    return str + ' [Default: %default]'


USAGE_STRING = """
  USAGE:      python trainPerceptron.py <options>
  EXAMPLES:   (1) python trainPerceptron.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python trainPerceptron.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAINING_SET_SIZE, type="int")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=20, type="int")
    parser.add_option('-l', '--learning_rate', help=default("Learning rate to use in training"), default=1, type="float")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-w', '--weights_path', help=default("Where to save your models weights to"), default="./models/q3_weights.model", type="string")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Training Perceptron")
    print("--------------------")

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.test <= 0:
        print("Testing set size should be a positive integer (you provided: %d)" % options.test)
        print(USAGE_STRING)
        sys.exit(2)

    args['num_iterations'] = options.iterations
    args['training_size'] = options.training
    args['testing_size'] = options.test
    args["learning_rate"] = options.learning_rate
    args['weights_path'] = options.weights_path

    print(args)
    print(options)

    return args, options


def runClassifier(args):

    numTraining = args['training_size']
    numTest = args['testing_size']
    num_iterations = args['num_iterations']
    learning_rate = args['learning_rate']
    weight_save_path = args['weights_path']

    df_train = pd.read_csv("./pacmandata/q3_train.csv").head(numTraining)
    df_validation = pd.read_csv("./pacmandata/q3_validation.csv").head(numTest)
    df_test = pd.read_csv("./pacmandata/q3_test.csv").head(numTest)

    print(len(df_train))
    print(len(df_validation))
    print(len(df_test))

    trainingData = df_train.drop(columns=["label"]).to_numpy()
    trainingLabels = df_train["label"]

    validationData = df_validation.drop(columns=["label"]).to_numpy()
    validationLabels = df_validation["label"]

    testData = df_test.drop(columns=["label"]).to_numpy()
    testLabels = df_test["label"]

    # create the model
    classifier = PerceptronPacman(num_train_iterations=num_iterations, learning_rate=learning_rate)

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)

    print("Testing...")
    test_performance = classifier.evaluate(testData, testLabels)
    print(test_performance)

    # Save the model weights to file
    classifier.save_weights(weight_save_path)

    return classifier


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args)