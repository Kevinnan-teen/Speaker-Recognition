import glob
import os
import numpy as np
import scipy.io.wavfile as wav

import shutil

def mkdir_files(path_read_folder, path_write_wav_file):

    target_dir = '../../speech/TIMIT/TRAIN/'
    target_dir2 = '../../speech/TIMIT/TEST/'

    drs = os.listdir(path_read_folder)
    s = 0
    for dr in drs:   
        # DR*
        drs_path = os.path.join(path_read_folder, dr)
        samples = os.listdir(drs_path)
        #print(len(samples))
        for sample in samples:
            samples_path = os.path.join(drs_path, sample)
            #print(samples_path)            
            for filename in glob.glob(os.path.join(samples_path, 'SX*_.wav')):
                #print(filename)
                s_file = filename.split('\\')
                mkdir_dir = target_dir + s_file[1] + '/' + s_file[2]
                # print(s_file[3])
                target = mkdir_dir + '/' + s_file[3]
                print(target)
                shutil.copy(filename, target)
                # if os.path.exists(mkdir_dir):
                #     pass
                # else:
                #     os.makedirs(mkdir_dir)

                # print(mkdir_dir)
                s += 1
                # break
                #print(len(s_file))
                #print(s_file[])
    # drs = os.listdir(target_dir)
    # s = 0
    # for dr in drs:   
    #     # DR*
    #     drs_path = os.path.join(target_dir, dr)
    #     samples = os.listdir(drs_path)
    #     #print(len(samples))
    #     for sample in samples:
    #         samples_path = os.path.join(drs_path, sample)
    #         #print(samples_path)            
    #         for filename in glob.glob(os.path.join(samples_path, 'SI*_.wav')):
    #             print(filename)
    #             # os.remove(filename)
    #             s_file = filename.split('\\')
    #             # print(s_file)
    #             # print(filename[25:28])
    #             target = target_dir2 + filename[25:28] + '/' + s_file[1] + '/' + s_file[2]
    #             print(target)
    #             # if os.path.exists(mkdir_dir):
    #             #     pass
    #             # else:
    #             #     os.makedirs(mkdir_dir)
    #             # s_file = filename.split('\\')
    #             # mkdir_dir = target_dir2 + s_file[1] + '/' + s_file[2]
    #             # # print(s_file[3])
    #             #target = mkdir_dir + '/' + s_file[3]
    #             #print(target)
    #             shutil.copy(filename, target)
    #             os.remove(filename)
    #             break
    print("sum:", s)


# 合并SX的个句子和SI的第一个句子作为训练集
# SI的后俩个个句子作为测试集
def merge_files(path_read_folder, path_write_wav_file):

    target_dir = '../../speech/TIMIT/TRAIN/'
    save_name = ''

    drs = os.listdir(target_dir)
    s = 0
    for dr in drs:   
        # DR*
        drs_path = os.path.join(target_dir, dr)
        samples = os.listdir(drs_path)
        #print(len(samples))

        for sample in samples:
            samples_path = os.path.join(drs_path, sample)
            #print(samples_path)
            merged_signal = []            
            for filename in glob.glob(os.path.join(samples_path, '*.wav')):
                #print(filename)
                s_file = filename.split('\\')
                save_name = target_dir + filename[25:28] + '/' + s_file[1] + '/' + "merge_result.wav"
                # print(save_name)
                sr, signal = wav.read(filename)
                merged_signal.append(signal)
            print(len(merged_signal))
            # print(merged_signal[0].shape, merged_signal[1].shape)
            merged_signal=np.hstack(merged_signal)
            merged_signal = np.asarray(merged_signal, dtype=np.int16)
            wav.write(save_name, sr, merged_signal)
    print("sum:", s)
            # print(sample)
        # merged_signal = []
        # for filename in glob.glob(os.path.join(path_read_folder, 'sentence*.wav')):
        #     print(filename)
            #sr, signal = wav.read(filename)
            #merged_signal.append(signal)
        # print(len(merged_signal))
        # print(merged_signal[0].shape, merged_signal[1].shape)
        # merged_signal=np.hstack(merged_signal)
        # merged_signal = np.asarray(merged_signal, dtype=np.int16)
        # wav.write(path_write_wav_file, sr, merged_signal)
 

#noisy train total
path_read_folder = '../../speech/TIMIT2/TRAIN'
path_write_wav_file = '../../speech/male/male_'
merge_files(path_read_folder, path_write_wav_file)
    

