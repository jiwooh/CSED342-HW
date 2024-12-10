import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """
    # BEGIN_YOUR_ANSWER
    return {"so": 0, "interesting": 0, "great": 1, "plot": 1, "bored": -1, "not": -1} # iterate w <- w + phi(x) * y for all datapoints
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    return collections.Counter(x.split())
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER
    for _ in range(0, numIters):
        for x, y in trainExamples:
            features = featureExtractor(x)
            prediction = sigmoid(sum(weights.get(fea, 0) * value for fea, value in features.items()))
            for fea, value in features.items():
                if y == 1:
                    gradient = value * (prediction - y)
                else: # y == -1
                    gradient = value * prediction
                weights[fea] = weights.get(fea, 0) - eta * gradient

    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    ngram = {}
    x = x.split()
    for i in range(0, len(x) - n + 1):
        block = " ".join(x[i:i+n])
        if block not in ngram:
            ngram[block] = 1
        else:
            ngram[block] += 1
    return ngram
    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        def sigmoid(n):
            return 1 / (1 + np.exp(-n))
        
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1 # z1 = x @ W1 + b1
        self.h = sigmoid(self.z1) # h = sigmoid(z1)
        self.z2 = np.dot(self.h, self.W2) + self.b2 # z2 = h @ W2 + b2
        self.pred = sigmoid(self.z2)
        return self.pred.reshape(-1)
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER
        return -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER

        pred = pred.reshape(-1, 1) # (4,1)
        target = target.reshape(-1, 1) # (4,1)

        d_pred = pred - target # (4,1)
        d_z2 = d_pred * pred * (1 - pred) # (4,1)

        d_W2 = np.dot(self.h.T, d_z2) # (16,1)
        d_b2 = np.sum(d_z2, axis=0) # (1,1)
        
        d_h = np.dot(d_z2, self.W2.T) # (4,16)
        d_z1 = d_h * self.h * (1 - self.h) # (4,16)
        
        d_W1 = np.dot(self.x.T, d_z1) # (2,16)
        d_b1 = np.sum(d_z1, axis=0) # (1,16)
        
        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"].reshape(1, -1)
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"].reshape(1, -1)
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER
        losses = []
        for epoch in range(epochs):
            for i in range(len(X)):
                pred = self.forward(X[i].reshape(1,-1))
                loss = self.loss(pred, Y[i])
                gradients = self.backward(pred, Y[i])
                self.update(gradients, learning_rate)
            losses.append(loss)
        return losses[-1].item()
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))