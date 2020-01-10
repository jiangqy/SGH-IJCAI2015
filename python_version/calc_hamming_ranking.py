# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: calc_hamming_ranking.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 20-1-10
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import multiprocessing as mp

def create_similarity(label_x, label_y):
    if isinstance(label_x, np.ndarray):
        label_x, label_y = label_x.squeeze(), label_y.squeeze()
        if label_x.ndim <= 1 and label_y.ndim <= 1:
            sim = (np.equal(label_x.reshape(-1, 1), label_y.reshape(1, -1))).astype(np.float32)
            return sim
        else:
            if label_x.ndim == 1:
                label_x = label_x.reshape(1, -1)
            if label_y.ndim == 1:
                label_y = label_y.reshape(1, -1)
            sim = (label_x.dot(label_y.T) > 0).astype(np.float32)
            return sim


def __hamming_dist__(b1, b2):
    if b1.ndim == 1:
        b1 = b1.reshape(1, -1)
    if b2.ndim == 1:
        b2 = b2.reshape(1, -1)
    bit = b1.shape[1]
    hamm = (bit - b1.dot(b2.T)) / 2.
    return hamm.squeeze()


class HammingRanking(object):
    def __init__(self, database_codes, database_labels, topk=None, verbose=None):
        super(HammingRanking, self).__init__()
        self.database_codes = database_codes
        self.database_labels = database_labels
        self.topk = topk

        self.input = mp.Queue()
        self.procs = []
        self.num_procs = 10

        aps = mp.Manager()
        self.aps = aps.list()
        if self.topk is not None:
            topkaps = mp.Manager()
            self.topkaps = topkaps.list()

        self.verbose = verbose
        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

        if self.verbose:
            logger.info('init done')

    def calc_ap(self, params):
        idx, testcode, testlabel = params[0], params[1], params[2]
        y_score = __hamming_dist__(testcode, self.database_codes).squeeze()
        y_true = create_similarity(testlabel, self.database_labels).squeeze()

        y_true = np.array(y_true)
        tsum = int(np.sum(y_true))
        if tsum == 0:
            return {'ap': 0, 'topkap': 0}

        y_score = np.array(y_score)
        ind = np.argsort(y_score)
        y_true = y_true[ind]
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(y_true == 1)) + 1.0
        ap = np.mean(count / tindex)
        topkap = 0.
        if self.topk is not None:
            ty_true = y_true[:self.topk]
            tsum = int(np.sum(ty_true))
            if tsum != 0:
                count = np.linspace(1, tsum, tsum)
                tindex = np.asarray(np.where(ty_true == 1)) + 1.0
                topkap = np.mean(count / tindex)
            if self.verbose:
                print('idx: {:4d}, ap: {:.4f}, top{}-ap: {:.4f}'.format(
                    idx, ap, self.topk, topkap
                ))
        else:
            if self.verbose:
                print('idx: {:4d}, ap: {:.4f}'.format(
                    idx, ap
                ))
        return {'ap': ap, 'topkap': topkap}

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                results = self.calc_ap(params)
                self.aps.append(results['ap'])
                if self.topk is not None:
                    self.topkaps.append(results['topkap'])
            except Exception as e:
                print('Exception: {}'.format(e))

    def start(self, test_codes, test_labels):
        num_test = test_codes.shape[0]
        for idx in range(num_test):
            self.input.put([idx, test_codes[idx], test_labels[idx]])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            if self.verbose:
                logger.info('process: {} done'.format(idx))

    def get_results(self):
        map = np.mean(np.array(list(self.aps)))
        if self.topk is not None:
            topkmap = np.mean(np.array(list(self.topkaps)))
            return {'map': map, 'topkmap': topkmap}
        return {'map': map}


def calc_hamming_ranking(test_codes, database_codes, test_labels, database_labels, topk=None, verbose=None):
    if test_labels.ndim == 1:
        test_labels, database_labels = test_labels.reshape(-1, 1), database_labels.reshape(-1, 1)
    hr = HammingRanking(database_codes, database_labels, topk, verbose)
    hr.start(test_codes, test_labels)
    hr.stop()
    results = hr.get_results()
    return results

