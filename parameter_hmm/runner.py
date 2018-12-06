import collections
import hashlib
import os
from urllib.parse import urlparse

import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib

"""
Trainer.
Use HMM to create profile helping detecting abnormal request.
This is just a simple demo which need enhance and tuning. 
"""


class Trainer:
    """
    Trainer using HMM to train the dataset and get profiles
    """

    def __init__(self):
        self.log = 'log_demo.txt'
        # a dict with the keys as hash(host+path+param) and the values as extracted sequence lists.
        self.data_dict = collections.defaultdict(list)
        # a dict with the keys as hash(host+path+param) and the values as profiles.
        self.profile_dict = dict()
        # scores for training data
        self.min_score, self.max_score = 0, 0

    def _read_data(self):
        """
        load data from log file and save into the data_dict.
        :return:
        """
        with open(self.log, 'r') as fp:
            for line in fp.readlines():
                items = line.strip().split()
                k = hashlib.md5((items[2] + items[3] + items[4]).encode()).hexdigest()
                self.data_dict[k].append(self._extract_item(items[5]))

    @staticmethod
    def _extract_item(item):
        """
        Transfer a string to extracted one.
        0-9 -> N (48-57) - 1
        a-zA-Z -> A (65-90, 97-122) - 2
        symbol and others -> c - 3
        :param item:
        :return:
        """
        res = []
        for i in range(0, len(item)):
            if 48 <= ord(item[i]) <= 57:
                res.append(1)
            elif 65 <= ord(item[i]) <= 90 or 97 <= ord(item[i]) <= 122:
                res.append(2)
            elif 0 <= ord(item[i]) <= 127:
                res.append(3)
            else:
                res.append(4)
        return res

    @staticmethod
    def single_train(x):
        """
        train one data list.
        :param x: list
        :return: HMM profile object
        """
        profile = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
        x_length = list()
        for i in x:
            x_length.append(len(i))
        x_data = np.array(np.concatenate(x)).reshape(-1, 1)
        profile.fit(x_data, x_length)
        return profile

    def fit(self):
        """
        load data, call other functions to generate profiles, and save them into the files.
        :return:
        """
        self._read_data()
        for k in self.data_dict:
            profile = self.single_train(x=self.data_dict[k])
            joblib.dump(profile, os.path.join('profiles', '{}.pkl'.format(k)))
            self.profile_dict[k] = profile

    def ranging(self):
        """
        find the range of scores for normal values.
        :return:
        """
        min_score, max_score = [], []
        with open(self.log, 'r') as fp:
            for line in fp.readlines():
                items = line.strip().split()
                k = hashlib.md5((items[2] + items[3] + items[4]).encode()).hexdigest()
                if os.path.exists(os.path.join('profiles', '{}.pkl'.format(k))):
                    profile = joblib.load(os.path.join('profiles', '{}.pkl'.format(k)))
                    v = np.array(self._extract_item(items[5])).reshape(-1, 1)
                    score = profile.score(v)
                    min_score.append(score)
                    min_score = [min(min_score)]
                    max_score.append(score)
                    max_score = [max(max_score)]
        with open('range.txt', 'w') as fp:
            fp.write('{} {}'.format(min_score[0], max_score[0]))

    def predict(self, url):
        """
        return score of a input URL. Be noticed that this is a simple version
        since this is not the focus to parse urls. Just use some simple URLs
        to help feeling the method.
        :param url:
        :return:
        """
        url_object = urlparse(url)
        k = hashlib.md5((url_object.netloc + url_object.path + url_object.query.split('=')[0]).encode()).hexdigest()
        if os.path.exists(os.path.join('profiles', '{}.pkl'.format(k))):
            profile = joblib.load(os.path.join('profiles', '{}.pkl'.format(k)))
            v = np.array(self._extract_item(url_object.query.split('=')[1])).reshape(-1, 1)
            score = profile.score(v)
            if os.path.exists('range.txt'):
                with open('range.txt', 'r') as fp:
                    min_score, max_score = map(float, fp.read().split())
                if min_score < score < max_score:
                    print('Normal')
                else:
                    print('Abnormal')
            return score
        else:
            raise Exception


def test():
    trainer = Trainer()
    # trainer.fit()
    # trainer.ranging()
    score = trainer.predict('www.xxx.com/index.php?id=<script></script>')
    print(score)


if __name__ == '__main__':
    test()
