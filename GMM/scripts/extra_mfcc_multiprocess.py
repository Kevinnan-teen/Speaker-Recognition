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
import glob
import multiprocessing


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

def extraMFCC(filename, savename, mfcc_num=13):
    nw = 320  #对于16KHz的文件，20ms的采样点个数
    inc = 160
    wave_data=getWaveData(filename)
    winfunc = signal.hann(nw)
    X = enframe(wave_data, nw, inc, winfunc)
    frameNum = X.shape[0]  # 返回矩阵列数，获取帧数

    data=[]
    for oneframe in  X:
        tmpList=list()
        mfccs = librosa.feature.mfcc(y=oneframe, sr=16000, n_mfcc=mfcc_num)
        # print(mfccs.shape)
        for a in mfccs:
            # print(a.shape)
            tmpList.append(a[0])
        data.append(tmpList)
    data=np.array(data)
    # data.shape : (frames_num, 24)
    print(data.shape)
    np.save(savename, data)


def extra_train_MFCC(start, end, spk_num):
    target_dir = '../speech/TIMIT/TRAIN/'

    drs = os.listdir(target_dir)

    i = 0
    print(drs)

    for dr in drs[start:end]:
        # DR*
        drs_path = os.path.join(target_dir, dr)
        samples = os.listdir(drs_path)
        #print(len(samples))
        for sample in samples:            
            samples_path = os.path.join(drs_path, sample)
            # print(samples_path)
            filename = os.path.join(samples_path, 'merge_result.wav')
            # print(filename)
            # if os.path.exists(filename):
            #     print(filename)
            spk_dir = '../speech/TIMIT/TRAIN_MFCC/spk_' + str(spk_num+1)            
            save_name = spk_dir  + '/spk_' + str(spk_num+1) + '_13d_mfcc.npy'
            print(save_name)
            # for filename in glob.glob(os.path.join(samples_path, '*.wav')):
            #   # print(filename)   
            #   s_file = filename.split('\\')
            #   # print(s_file[2][:-4])
            #   spk_dir = '../speech/TIMIT/TEST_MFCC/' + 'spk_' + str(i+1)
            #   save_name = spk_dir + '/' + s_file[2][:-4] + 'mfcc.npy'
            extraMFCC(filename, save_name, 13)
            #   print(save_name)
            # i += 1
            spk_num += 1


def extra_test_MFCC(start, end, spk_num):
    target_dir = '../speech/TIMIT/TEST/'

    drs = os.listdir(target_dir)

    i = 0

    for dr in drs[start:end]:
      # DR*
      drs_path = os.path.join(target_dir, dr)
      samples = os.listdir(drs_path)
      #print(len(samples))

      for sample in samples:
          samples_path = os.path.join(drs_path, sample)
          for filename in glob.glob(os.path.join(samples_path, '*.wav')):
              # print(filename)
              s_file = filename.split('\\')
              # print(s_file[2][:-4])
              spk_dir = '../speech/TIMIT/TEST_MFCC/' + 'spk_' + str(spk_num+1)
              save_name = spk_dir + '/spk_' + str(spk_num+1) + '_13d_mfcc.npy'
              extraMFCC(filename, save_name)
              print(save_name)
          spk_num += 1


def train_multiprocess():
    extra_mfcc = [multiprocessing.Process(target=extra_train_MFCC, args=(0, 2, 0,),),
                             multiprocessing.Process(target=extra_train_MFCC, args=(2, 4, 114,),),
                             multiprocessing.Process(target=extra_train_MFCC, args=(4, 6, 258,),),
                             multiprocessing.Process(target=extra_train_MFCC, args=(6, 8, 363),)]
                                        
            
    for process in extra_mfcc:
        process.daemon = True
        process.start()
    for process in extra_mfcc:
        process.join()


def test_multiprocess():
    extra_mfcc = [multiprocessing.Process(target=extra_test_MFCC, args=(0, 2, 0,),),
                             multiprocessing.Process(target=extra_test_MFCC, args=(2, 4, 114,),),
                             multiprocessing.Process(target=extra_test_MFCC, args=(4, 6, 258,),),
                             multiprocessing.Process(target=extra_test_MFCC, args=(6, 8, 363),)]
                                        
            
    for process in extra_mfcc:
        process.daemon = True
        process.start()
    for process in extra_mfcc:
        process.join()

if __name__ == '__main__':
    train_multiprocess()
   