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
import multiprocessing


def getGMM(filename):

	data = np.load(filename)
	# 高斯混合模型 5个样本时，聚类数量为3，效果最好
	# 使用贝叶斯GMM，可避免数量选择
	gmm = GaussianMixture(7, covariance_type='diag', random_state=0).fit(data)
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



def train_gmm_model(start_num, end_num):
    start=time.clock()
    # 训练gmm，只需修改（1）
    root_dir = '../models/TIMIT_MFCC_24/'
    GMMs=[]
    # female
    for i in range(start_num, end_num):
    	spk_mfcc = '../speech/TIMIT/TRAIN_MFCC/' + 'spk_' + str(i+1) + '/spk_' + str(i+1) + '_13d_mfcc.npy'  
    	if os.path.exists(spk_mfcc):
            # 修改（2）
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
    # 修改（1）
    root_dir = '../models/TIMIT_MFCC_24/'

    for i in range(462):
        # mfcc_npy_file = '../speech/TIMIT/TEST_MFCC/' + 'spk_' + str(i+1) + '/' + s_file[2][:-4] + 'mfcc.npy'
        test_mfcc_path = '../speech/TIMIT/TEST_MFCC/' + 'spk_' + str(i+1) + '/'
        if os.path.exists(test_mfcc_path):        	
        	mfcc_npy_file = test_mfcc_path + os.listdir(test_mfcc_path)[0]
            # mfcc_npy_file = test_mfcc_path + '/spk_' + str(i+1) + '_13d_mfcc.npy'
        # 加载模型
        # 修改（2）
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


def gmm_train_multiprocess():
    train_mfcc = [multiprocessing.Process(target=train_gmm_model, args=(0, 100,)),
                             multiprocessing.Process(target=train_gmm_model, args=(100,200,)),
                             multiprocessing.Process(target=train_gmm_model, args=(200,300,)),
                             multiprocessing.Process(target=train_gmm_model, args=(300,400,)),
                             multiprocessing.Process(target=train_gmm_model, args=(400,462,))]
                                
    for process in train_mfcc:
        process.daemon = True
        process.start()
    for process in train_mfcc:
        process.join()

if __name__=="__main__":    
    gmm_train_multiprocess()

