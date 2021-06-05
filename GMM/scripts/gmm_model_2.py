# _*_ coding=utf-8 _*_
from scipy import signal
import pylab as pl
from sklearn.mixture import GaussianMixture
import joblib
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import wave
import time
import math


def getGMM(filename):

    y, sr = librosa.load(filename)
    # 提取 MFCC feature
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=160, win_length=240)
    print(mfccs.shape)

    # 高斯混合模型 5个样本时，聚类数量为3，效果最好
    # 使用贝叶斯GMM，可避免数量选择
    gmm = GaussianMixture(1, covariance_type='full', random_state=0).fit(mfccs.T)
    return gmm


def softmax(scores):
    ss = 0.0
    Sum = 0.0
    for score in scores:
        ss += score
        
    scores = [(-1)*float(i)/ss for i in scores]

    for score in scores:
        Sum += math.exp(score)

    # for score in scores:        
    print("probalitiy:{0}, index:{1}".format(math.exp(max(scores)) / Sum, scores.index(max(scores))))


def train_gmm_model():
    start=time.clock()
    female_siri_path = '../speech/female/female_'

    for i in range(9):
        train_female_siri_file = female_siri_path + str(i+1) + '/siri_1.wav'
        model_file = '../models/direct_femele_' + str(i+1) + '_gmm.model'
        female_siri_gmm = getGMM(train_female_siri_file)
        joblib.dump(female_siri_gmm, model_file)
        print("finished")
        # model = joblib.load(model_file)
    timePointAfterGmm=time.clock()


def test_gmm_model():
    female_siri_path = '../speech/female/female_'
    #对采样信号处理
    nw=320
    inc = 160
    winfunc = signal.hann(nw)

    test_data_list = []
    gmm_model_list = []
    for i in range(9):
        test_female_siri_file = female_siri_path + str(i+1) + '/siri_2.wav'
        # 加载模型
        model_file = '../models/direct_femele_' + str(i+1) + '_gmm.model'
        gmm_model = joblib.load(model_file)
        gmm_model_list.append(gmm_model)

        y, sr = librosa.load(test_female_siri_file)
        # 提取 MFCC feature
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=160, win_length=240)

        test_data_list.append(mfccs.T)
        # maxPro=GMMs[0].score(data)

    # 测试
    for model in gmm_model_list:
        scores = []
        for test_data in test_data_list:
            test_score = model.score(test_data)
            scores.append(test_score)
            print("test_score:", test_score)
        softmax(scores)
        print("-------------------")

if __name__=="__main__":
    train_gmm_model()
    test_gmm_model()

