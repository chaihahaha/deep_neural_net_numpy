import numpy as np


def linear(matrix):
    return matrix


def leakyReLU(matrix):
    return np.where(matrix < 0, 0.01 * matrix, matrix)


def d_leakyReLU(matrix):
    return np.where(matrix >= 0, 1, 0.01)


def ReLU(matrix):
    return np.where(matrix < 0, 0, matrix)


def d_ReLU(matrix):
    return np.where(matrix >= 0, 1, 0)


def sigmoid(matrix):
    return 1/(1 + np.exp(-matrix))


def d_sigmoid(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))


def loss(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat), axis=1))


def d_loss(y, y_hat):
    # y_hat：网络输出  y：正确输出
    return -y/y_hat + (1 - y)/(1 - y_hat)
