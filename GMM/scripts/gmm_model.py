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


def enframe(wave_data, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    wave_data:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    wlen=len(wave_data) #信号总长度
    if wlen<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*wlen-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-wlen,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((wave_data,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

def getWaveData(filename):
    fw = wave.open(filename,'rb')
    params = fw.getparams()
    #print(params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = fw.readframes(nframes)
    wave_data = np.fromstring(str_data, dtype=np.int16)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # wave幅值归一化
    fw.close()
    return wave_data

def getGMM(filename):

    nw = 320  #对于16KHz的文件，20ms的采样点个数
    inc = 160
    wave_data=getWaveData(filename)
    winfunc = signal.hann(nw)
    X = enframe(wave_data, nw, inc, winfunc)
    frameNum = X.shape[0]  # 返回矩阵列数，获取帧数

    data=[]
    for oneframe in  X:
        tmpList=list()
        mfccs = librosa.feature.mfcc(y=oneframe, sr=16000, n_mfcc=24)
        # print(mfccs.shape)
        for a in mfccs:
            # print(a.shape)
            tmpList.append(a[0])
        data.append(tmpList)
    data=np.array(data)
    # data.shape : (frames_num, 24)
    print(data.shape)

    # 高斯混合模型 5个样本时，聚类数量为3，效果最好
    # 使用贝叶斯GMM，可避免数量选择
    gmm = GaussianMixture(3, covariance_type='full', random_state=0).fit(data)
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
    male_siri_path = '../speech/male/male_'
    GMMs=[]
    # female
    for i in range(9):
        train_female_siri_file = female_siri_path + str(i+1) + '/sentence_1.wav'
        model_file = '../models/sentence_female_' + str(i+1) + '_gmm.model'
        female_siri_gmm = getGMM(train_female_siri_file)
        joblib.dump(female_siri_gmm, model_file)
        print("finished")
    # male
    for i in range(6):
        train_male_siri_file = male_siri_path + str(i+1) + '/sentence_1.wav'
        model_file = '../models/sentence_male_' + str(i+1) + '_gmm.model'
        male_siri_gmm = getGMM(train_male_siri_file)
        joblib.dump(male_siri_gmm, model_file)
        print("finished")
        # model = joblib.load(model_file)
    timePointAfterGmm=time.clock()


def test_gmm_model():
    female_siri_path = '../speech/female/female_'
    male_siri_path = '../speech/male/male_'
    #对采样信号处理
    nw=320
    inc = 160
    winfunc = signal.hann(nw)

    test_data_list = []
    gmm_model_list = []
    # female
    for i in range(9):
        test_female_siri_file = female_siri_path + str(i+1) + '/xiaoai_2.wav'
        # 加载模型
        model_file = '../models/sentence_female_' + str(i+1) + '_gmm.model'
        gmm_model = joblib.load(model_file)
        gmm_model_list.append(gmm_model)

        testFrames=enframe(getWaveData(test_female_siri_file), nw, inc, winfunc)
        data=[]
        # 提取测试序列MFCC特征
        for oneframe in testFrames:
            tmpList=list()
            for a in librosa.feature.mfcc(y=oneframe, sr=16000 , n_mfcc=24):
                tmpList.append(a[0])
            data.append(tmpList)
        data=np.array(data)
        print(data.shape)
        test_data_list.append(data)
        # maxPro=GMMs[0].score(data)
    # male
    for i in range(6):
        test_male_siri_file = male_siri_path + str(i+1) + '/xiaoai_2.wav'
        # 加载模型
        model_file = '../models/sentence_male_' + str(i+1) + '_gmm.model'
        gmm_model = joblib.load(model_file)
        gmm_model_list.append(gmm_model)

        testFrames=enframe(getWaveData(test_male_siri_file), nw, inc, winfunc)
        data=[]
        # 提取测试序列MFCC特征
        for oneframe in testFrames:
            tmpList=list()
            for a in librosa.feature.mfcc(y=oneframe, sr=16000 , n_mfcc=24):
                tmpList.append(a[0])
            data.append(tmpList)
        data=np.array(data)
        print(data.shape)
        test_data_list.append(data)
    # 测试
    for model in gmm_model_list:
        scores = []
        for test_data in test_data_list:
            test_score = model.score(test_data)
            ss = model.score_samples(test_data)
            scores.append(test_score)
            # print("test_score:", ss.shape)
        softmax(scores)
        print("-------------------")



if __name__=="__main__":
    train_gmm_model()
    test_gmm_model()

