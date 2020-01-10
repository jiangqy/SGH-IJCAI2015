# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: itq_demo.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 20-1-10
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import h5py

import numpy as np

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
    print('load data done')

    mean_ = np.mean(database_features, axis=0)
    database_features -= mean_
    test_features -= mean_

    n_iter = 50
    print('start training')
    Wx = learn_hash_function(bit, n_iter, database_features)
    database_codes = np.sign(database_features.dot(Wx))
    test_codes = np.sign(test_features.dot(Wx))

    print('start evaluating')
    results = calc_hamming_ranking(test_codes, database_codes, test_labels, database_labels)
    print('map: {:.4f}'.format(results['map']))



def learn_hash_function(bit, n_iter, features):
    c = np.cov(features.transpose())
    l, pc = np.linalg.eig(c)

    l_pc_ordered = sorted(zip(l, pc.transpose()), key=lambda _p: _p[0], reverse=True)
    pc_top = np.array([p[1] for p in l_pc_ordered[:bit]]).transpose()

    v = np.dot(features, pc_top)

    b, rotation = itq_rotation(bit, v, n_iter)

    proj = np.dot(pc_top, rotation)
    return proj


def itq_rotation(bit, v, n_iter):
    r = np.random.randn(bit, bit)
    u11, s2, v2 = np.linalg.svd(r)
    r = u11[:, :bit]

    for i in range(n_iter):
        z = np.dot(v, r)
        ux = np.ones(z.shape) * -1.
        ux[z >= 0] = 1
        c = np.dot(ux.transpose(), v)
        ub, sigma, ua = np.linalg.svd(c)
        r = np.dot(ua, ub.transpose())

    z = np.dot(v, r)
    b = np.ones(z.shape) * -1
    b[z >= 0] = 1
    return b, r


if __name__ == "__main__":
    main()

'''bash
python itq_demo.py
'''