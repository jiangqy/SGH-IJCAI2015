# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: sgh_demo.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 20-1-10
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import h5py

import numpy as np
import scipy.linalg as alg

from calc_hamming_ranking import calc_hamming_ranking


def load_data():
    filename = 'MNIST.h5'
    fo = h5py.File(filename, mode='r')
    database_features = fo['XDatabase'][()].astype(np.float32)
    test_features = fo['XTest'][()].astype(np.float32)
    database_labels = fo['databaseL'][()].astype(np.float32)
    test_labels = fo['testL'][()].astype(np.float32)

    return database_features, test_features, database_labels, test_labels


def main():
    bit = 64
    database_features, test_features, database_labels, test_labels = load_data()

    mean_ = np.mean(database_features, axis=0)
    database_features -= mean_
    test_features -= mean_

    database_norm = np.linalg.norm(database_features, axis=1, keepdims=True)
    database_features /= database_norm

    test_norm = np.linalg.norm(test_features, axis=1, keepdims=True)
    test_features /= test_norm

    print('load data done')

    num_kernel = 300
    num_data = database_features.shape[0]
    index = np.random.permutation(range(1,num_data))[: num_kernel]
    bases = database_features[index]

    print('start training')
    database_codes, test_codes = trainSGH(bases, database_features, test_features, bit)

    print('start evaluating')
    results = calc_hamming_ranking(test_codes, database_codes, test_labels, database_labels)
    print('map: {:.4f}'.format(results['map']))


def trainSGH(bases, train_features, test_features, bit):
    num_train = train_features.shape[0]
    rho = 2

    # construct PX and QX
    e = np.exp(1)
    norm_ = np.linalg.norm(train_features, axis=1, keepdims=True) ** 2
    norm_ = np.exp(-norm_ / rho)

    alpha = np.sqrt(2* (e*e - 1) / (e*rho))
    beta = np.sqrt((e*e+1) / e)

    part2 = beta * norm_
    part1 = alpha * norm_ * train_features
    px = np.c_[part1, part2, np.ones((num_train, 1), dtype=np.float32)]
    qx = np.c_[part1, part2, -1 * np.ones((num_train, 1), dtype=np.float32)]

    # construct kernel for training set
    kernel_train = dist_matrix(train_features, bases)
    delta = np.mean(kernel_train)
    kernel_train = np.exp(- kernel_train / (2 * delta))
    mean_ = np.mean(kernel_train, axis=0)
    kernel_train -= mean_

    # training
    Wx = trainSGH_seq(kernel_train, px, qx, bit)

    database_codes = np.sign(kernel_train.dot(Wx))


    # construct kernel features for test
    kernel_test = dist_matrix(test_features, bases)
    kernel_test = np.exp(- kernel_test / (2 * delta))
    kernel_test -= mean_

    test_codes = np.sign(kernel_test.dot(Wx))

    return database_codes, test_codes


def trainSGH_seq(X, PX, QX, bit):
    ndim = X.shape[1]
    gamma = 1e-6
    A1 = bit * np.dot(X.T, PX).dot(np.dot(X.T, QX).T)

    Z = np.dot(X.T, X) + gamma * np.eye(ndim)

    Wx = np.random.randn(ndim, bit)



    for i in range(bit):
        eigvalues, eigvectors = alg.eigh(A1, Z, eigvals=(ndim-1, ndim-1))
        wx = eigvectors
        vx = np.dot(X.T, np.sign(np.dot(X, wx)))
        A1 = A1 - np.dot(vx, vx.T)
        Wx[:, i] = wx.squeeze()

    index = np.random.permutation(bit)
    iter = 1
    for j in range(iter):
        for i in range(bit):
            wx = Wx[:, index[i]].reshape(-1, 1)
            vx = np.dot(X.T, np.sign(np.dot(X, wx)))
            A1 = A1 + np.dot(vx, vx.T)
            eigvalues, eigvectors = alg.eigh(A1, Z, eigvals=(ndim - 1, ndim - 1))
            wx = eigvectors
            vx = np.dot(X.T, np.sign(np.dot(X, wx)))
            A1 = A1 - np.dot(vx, vx.T)
            Wx[:, index[i]] = wx.squeeze()
    return Wx


def dist_matrix(x, y):
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    assert colx == coly

    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    return x2 + y2 - 2 * xy


if __name__ == "__main__":
    main()


'''bash
python sgh_demo.py
'''