
# 使用保存的MFCC。

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

	data = np.load(filename)
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
    return scores.index(max(scores))



def train_gmm_model():
    start=time.clock()
    root_dir = '../models/TIMIT_MFCC_24/'
    GMMs=[]
    # female
    for i in range(462):
    	spk_mfcc = '../speech/TIMIT/TRAIN_MFCC/' + 'spk_' + str(i+1) + '/spk_' + str(i+1) + '_mfcc.npy'  
    	if os.path.exists(spk_mfcc):
    		model_file = root_dir + 'spk_' + str(i+1) + '/' + 'TIMIT_MFCC_24_gmm.model'
    		#print(model_file)
    		timit_gmm = getGMM(spk_mfcc)
    		joblib.dump(timit_gmm, model_file)
    		print("finished")
    	else:
    		print("nor exists")
        # save_name = '../mfcc_features/female_' + str(i+1) + '.npy'
        
        
    timePointAfterGmm=time.clock()


def test_gmm_model():
    test_data_list = []
    gmm_model_list = []
    root_dir = '../models/TIMIT_MFCC_24/'

    for i in range(462):
        # mfcc_npy_file = '../speech/TIMIT/TEST_MFCC/' + 'spk_' + str(i+1) + '/' + s_file[2][:-4] + 'mfcc.npy'
        test_mfcc_path = '../speech/TIMIT/TEST_MFCC/' + 'spk_' + str(i+1) + '/'
        if os.path.exists(test_mfcc_path):        	
        	mfcc_npy_file = test_mfcc_path + os.listdir(test_mfcc_path)[0]
        # 加载模型
        model_file = root_dir + 'spk_' + str(i+1) + '/' + 'TIMIT_MFCC_24_gmm.model'
        gmm_model = joblib.load(model_file)
        gmm_model_list.append(gmm_model)

        data = np.load(mfcc_npy_file)
        test_data_list.append(data)
    # 测试
    test_right = 0
    i = 0
    for model in gmm_model_list:
        scores = []
        for test_data in test_data_list:
            test_score = model.score(test_data)
            # ss = model.score_samples(test_data)
            scores.append(test_score)
            # print("test_score:", ss.shape)
        result = softmax(scores)
        if(i == result):
        	test_right += 1
        i += 1
        print("-------------------")
    print("right:{0}, accuracy:{1}".format(test_right, test_right/462))



if __name__=="__main__":
    #train_gmm_model()
    test_gmm_model()

